import importlib
import os
import re
import warnings
from collections import deque

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import v2 as transforms

from ..nn.model import (
    MultiPathRegressionNet,
    ClassificationNet,
    ClassificationNetRNN,
    RegressionNet,
    RegressionNetRNN,
    SegmentationNet,
    SegmentationNetRNN,
)
from .autocrop import Autocropper
from .common import to_scaled_tensor
from .postprocessing import (
    classifications_to_rails,
    regression_to_rails,
    scale_mask,
    scale_rails,
)


def parse_smoothing(smoothing):
    """Parses a smoothing specification into a (mode, param) pair.

    Args:
        smoothing (str or None): One of None, "rnn", "none", "boxcar<K>" or "ema<A>".

    Returns:
        tuple: (mode, param) where mode is None, "rnn", "none", "boxcar" or "ema",
            and param is the window size K (int) for boxcar, the alpha (float) for ema,
            and None otherwise.
    """
    if smoothing is None:
        return None, None
    if not isinstance(smoothing, str):
        raise ValueError(f"Invalid smoothing specification: {smoothing!r}")
    spec = smoothing.strip().lower()
    if spec in ("rnn", "none"):
        return spec, None
    match = re.fullmatch(r"boxcar(\d+)", spec)
    if match is not None:
        k = int(match.group(1))
        if k < 1:
            raise ValueError(f"Invalid boxcar window size: {smoothing!r} (must be >= 1)")
        return "boxcar", k
    match = re.fullmatch(r"ema(\d*\.?\d+)", spec)
    if match is not None:
        alpha = float(match.group(1))
        if not 0 < alpha <= 1:
            raise ValueError(f"Invalid ema alpha: {smoothing!r} (must be in ]0, 1])")
        return "ema", alpha
    raise ValueError(
        f"Invalid smoothing specification: {smoothing!r}"
        + " (expected None, 'rnn', 'none', 'boxcar<K>' or 'ema<A>')"
    )


class Detector:
    def __init__(self, model_path, crop_coords, runtime, device, *, smoothing=None):
        """Interface to infer the train ego-path detection model using PyTorch or TensorRT.

        Args:
            model_path (str): Path to the trained model directory (containing config.yaml and best.pt)
            crop_coords (tuple or str or None): Coordinates to use for cropping the input image before inference:
                - If tuple, should be the inclusive absolute coordinates (xleft, ytop, xright, ybottom) of the fixed region.
                - If str, should be "auto" to use automatic cropping.
                - If None, no cropping is performed.
            runtime (str): Runtime to use for model inference ("pytorch" or "tensorrt").
            device (str): Device to use for model inference ("cpu", "cuda", "cuda:x" or "mps").
            smoothing (str or None): Temporal smoothing applied to the raw prediction vector
                (before decoding), keyword-only. Defaults to None (legacy behavior):
                - None: temporal models use the RNN refinement, non-temporal models use the raw per-frame prediction.
                - "rnn": explicitly the RNN refinement path (temporal models only, raises ValueError otherwise).
                - "none": force the raw per-frame prediction (for a temporal model, the base network output, unrefined).
                - "boxcar<K>" (e.g. "boxcar5"): causal mean of the last K raw base predictions
                  (fewer than K buffered frames -> mean over what is available, so the first frame is the identity).
                - "ema<A>" (e.g. "ema0.5"): out[t] = A * x[t] + (1 - A) * out[t-1], with out[0] = x[0].
                Smoothing is applied to the raw prediction, which is expressed in coordinates
                relative to the current crop window; buffered predictions are therefore
                re-projected into the current crop's frame before averaging (see
                reproject_pred), which matters when crop_coords="auto" moves the window
                every frame. boxcar/ema are unsupported for the segmentation method, and for
                the classification method with crop_coords="auto" (no re-projection): a
                warning is emitted and the legacy (None) behavior is used instead.
                Call reset_temporal() between independent sequences to clear the smoothing state.
        """
        self.model_path = model_path
        self.runtime = runtime
        self.device = torch.device(device)
        with open(os.path.join(self.model_path, "config.yaml")) as f:
            self.config = yaml.safe_load(f)
        self.temporal = self.config.get("temporal", False)
        self.n_hypotheses = int(self.config.get("n_hypotheses", 1))
        if self.temporal:
            self.seq_len = self.config["seq_len"]
            self.path_buffer = deque(maxlen=self.seq_len)
        if isinstance(crop_coords, tuple) and len(crop_coords) == 4:
            self.crop_coords = crop_coords
        elif crop_coords == "auto":
            self.crop_coords = Autocropper(self.config)
        else:
            self.crop_coords = None

        self.smoothing = smoothing
        self.smoothing_mode, self.smoothing_param = parse_smoothing(smoothing)
        if self.smoothing_mode == "rnn" and not self.temporal:
            raise ValueError("smoothing='rnn' requires a temporal (RNN) model")
        self.drop_unsupported_smoothing()
        self.smoothing_buffer = (
            deque(maxlen=self.smoothing_param)
            if self.smoothing_mode == "boxcar"
            else None
        )
        self.smoothing_state = None  # (running output, its crop coords) of the ema

        if self.runtime == "pytorch":
            self.model = self.init_model_pytorch()
        elif self.runtime == "tensorrt":
            # lazy imports
            self.trt = importlib.import_module("tensorrt")
            self.cuda = importlib.import_module("pycuda.driver")
            os.environ["CUDA_MODULE_LOADING"] = "LAZY"
            # convert model to tensorrt if not already done
            if not os.path.exists(os.path.join(self.model_path, "best.trt")):
                self.convert_to_tensorrt()
            # init cuda context on device
            self.cuda.init()
            device = 0 if device == "cuda" else int(device.split(":")[-1])
            self.ctx = self.cuda.Device(device).retain_primary_context()
            self.ctx.push()
            self.exectx, self.bindings, self.shapes = self.init_model_tensorrt()
        else:
            raise ValueError

    def __del__(self):
        if self.runtime == "tensorrt":
            self.ctx.pop()

    def get_crop_coords(self):
        return (
            self.crop_coords()
            if isinstance(self.crop_coords, Autocropper)
            else self.crop_coords
        )

    def init_model_pytorch(self):
        if self.temporal:
            model = self.init_temporal_model_pytorch()
        elif self.config["method"] == "classification":
            model = ClassificationNet(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                classes=self.config["classes"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
            )
        elif self.config["method"] == "regression" and self.n_hypotheses > 1:
            model = MultiPathRegressionNet(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
                n_hypotheses=self.n_hypotheses,
            )
        elif self.config["method"] == "regression":
            model = RegressionNet(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
            )
        elif self.config["method"] == "segmentation":
            model = SegmentationNet(
                backbone=self.config["backbone"],
                decoder_channels=tuple(self.config["decoder_channels"]),
            )
        model.to(self.device).eval()
        state = torch.load(
            os.path.join(self.model_path, "best.pt"), map_location=self.device
        )
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                 for k, v in state.items()}
        model.load_state_dict(state)
        return model

    def init_temporal_model_pytorch(self):
        if self.config["method"] == "classification":
            return ClassificationNetRNN(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                classes=self.config["classes"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
                seq_len=self.config["seq_len"],
                rnn_hidden=self.config["rnn_hidden"],
                rnn_layers=self.config["rnn_layers"],
            )
        elif self.config["method"] == "regression":
            return RegressionNetRNN(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
                seq_len=self.config["seq_len"],
                rnn_hidden=self.config["rnn_hidden"],
                rnn_layers=self.config["rnn_layers"],
            )
        elif self.config["method"] == "segmentation":
            return SegmentationNetRNN(
                backbone=self.config["backbone"],
                decoder_channels=tuple(self.config["decoder_channels"]),
                seq_len=self.config["seq_len"],
                rnn_hidden=self.config["seg_rnn_hidden"],
                rnn_layers=self.config["rnn_layers"],
            )

    def drop_unsupported_smoothing(self):
        """Disables boxcar/ema smoothing when this method+crop cannot support it.

        Smoothing averages raw predictions, which live in coordinates relative to
        the current crop window (see reproject_pred). Re-projecting them into a
        common frame is only implemented for the regression method, so with a
        moving (auto) crop the other methods would average mismatched frames --
        and their outputs do not even share a shape with the regression vector,
        so averaging them raises rather than degrading quietly.

        Called from __init__, but exposed because ``crop_coords`` may be assigned
        after construction (callers that cache one detector across crop modes):
        the answer depends on the crop, so it has to be re-asked when it changes.
        """
        if self.smoothing_mode not in ("boxcar", "ema"):
            return
        unsupported = None
        if self.config["method"] == "segmentation":
            unsupported = "the segmentation method"
        elif self.config["method"] != "regression" and isinstance(
            self.crop_coords, Autocropper
        ):
            unsupported = f"method={self.config['method']!r} with crop_coords='auto'"
        if unsupported is not None:
            warnings.warn(
                f"smoothing={self.smoothing!r} is not supported for {unsupported},"
                + " falling back to no smoothing",
                stacklevel=2,
            )
            self.smoothing_mode, self.smoothing_param = None, None
            self.smoothing_buffer = None
            self.smoothing_state = None

    def reset_temporal(self):
        """Clears the stored ego-path history and the smoothing state
        (call between independent sequences/images)."""
        if self.temporal:
            self.path_buffer.clear()
        if self.smoothing_buffer is not None:
            self.smoothing_buffer.clear()
        self.smoothing_state = None

    # alias, for callers that do not care about the temporal/smoothing distinction
    reset = reset_temporal

    def reproject_pred(self, pred, src_crop, dst_crop):
        """Re-expresses a raw regression prediction from one crop's frame into another's.

        The raw prediction is in coordinates relative to the crop window it was computed
        on (x in [0, 1] of the crop width, anchors on a fixed y grid of the crop height).
        When the Autocropper moves the window between frames, buffered predictions live in
        different frames of reference and averaging them mixes coordinate systems. This
        maps `pred` onto the anchor grid of `dst_crop` so the average is taken in a single,
        current frame of reference.

        Measured on the labeled OSDaR23 sequences (experiments/eval_smoothing_crop.py,
        chromatic-laughter-5RNN, GT IoU through the full detect() path): re-projection
        recovers roughly half of the loss that a moving crop inflicts on boxcar5
        (crop="auto": 0.5540 -> 0.5627 against 0.5698 unsmoothed). Smoothing still only
        pays off with a static frame of reference (crop=None: 0.5929 -> 0.6110), because
        the Autocropper's own running average already low-pass filters the prediction;
        this is why detect.py defaults to --smoothing none.

        Args:
            pred (numpy.ndarray): Raw prediction, shape (1, 2 * anchors + 1).
            src_crop (tuple or None): Crop coordinates the prediction was computed on.
            dst_crop (tuple or None): Crop coordinates to re-express it in.

        Returns:
            numpy.ndarray: The prediction in `dst_crop`'s frame (unchanged if the crops match).
        """
        if src_crop == dst_crop or src_crop is None or dst_crop is None:
            return pred  # static frame of reference, nothing to re-project
        xs0, ys0, xs1, ys1 = src_crop
        xd0, yd0, xd1, yd1 = dst_crop
        ws, hs = xs1 - xs0, ys1 - ys0
        wd, hd = xd1 - xd0, yd1 - yd0
        if ws <= 0 or hs <= 0 or wd <= 0 or hd <= 0:
            return pred
        anchors = self.config["anchors"]
        traj = pred[0, :-1].reshape(2, anchors)
        yrel = np.linspace(1, 0, anchors)  # anchor grid, same in both frames
        # destination anchors -> absolute -> source-relative (clamped by np.interp)
        ysrc = ((yrel * hd + yd0) - ys0) / hs
        out = np.empty_like(pred)
        for i in range(2):
            xabs = np.interp(ysrc, yrel[::-1], traj[i, ::-1]) * ws + xs0
            out[0, i * anchors : (i + 1) * anchors] = (xabs - xd0) / wd
        # ylim is the fraction of the crop height covered from the bottom
        ylim = 1 / (1 + np.exp(-pred[0, -1]))
        ylim = 1 - (((1 - ylim) * hs + ys0) - yd0) / hd
        ylim = np.clip(ylim, 1e-6, 1 - 1e-6)
        out[0, -1] = np.log(ylim / (1 - ylim))
        return out

    def apply_smoothing(self, pred, crop_coords=None):
        """Applies the configured temporal smoothing to a raw prediction array.

        Buffered predictions are re-projected into `crop_coords`' frame of reference
        before averaging, so a moving (auto) crop does not corrupt the average.
        No-op for the None/"rnn"/"none" modes (those are handled upstream by
        selecting the base or the refined prediction).
        """
        if self.smoothing_mode == "boxcar":
            self.smoothing_buffer.append((np.asarray(pred, dtype=np.float32), crop_coords))
            buf = [self.reproject_pred(p, c, crop_coords) for p, c in self.smoothing_buffer]
            return np.mean(np.stack(buf), axis=0)
        elif self.smoothing_mode == "ema":
            pred = np.asarray(pred, dtype=np.float32)
            alpha = self.smoothing_param
            if self.smoothing_state is None:
                out = pred
            else:
                prev, prev_crop = self.smoothing_state
                prev = self.reproject_pred(prev, prev_crop, crop_coords)
                out = alpha * pred + (1 - alpha) * prev
            self.smoothing_state = (out, crop_coords)
            return out
        return pred

    def init_model_tensorrt(self):
        runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        with open(os.path.join(self.model_path, "best.trt"), "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        exectx = engine.create_execution_context()
        shapes = tuple(
            [tuple(engine.get_binding_shape(i)) for i in range(engine.num_bindings)]
        )
        bindings = [
            self.cuda.mem_alloc(np.prod(shape).item() * np.dtype(np.float32).itemsize)
            for shape in shapes
        ]
        return exectx, bindings, shapes

    def convert_to_tensorrt(self, precision="fp16"):
        pytorch_model = self.init_model_pytorch()
        dummy_input = torch.rand((1, *self.config["input_shape"])).to(self.device)
        torch.onnx.export(pytorch_model, dummy_input, "temp.onnx")
        trt_logger = self.trt.Logger(self.trt.Logger.ERROR)
        builder = self.trt.Builder(trt_logger)
        flag = 1 << (int)(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        config = builder.create_builder_config()
        parser = self.trt.OnnxParser(network, trt_logger)
        with open("temp.onnx", "rb") as model:
            parser.parse(model.read())
        if precision == "fp16":
            config.set_flag(self.trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)
        with open(os.path.join(self.model_path, "best.trt"), "wb") as f:
            f.write(engine.serialize())
        os.remove("temp.onnx")

    def infer_model_pytorch(self, img):
        tensor = to_scaled_tensor(img).unsqueeze(0).to(self.device)
        tensor = transforms.Resize(self.config["input_shape"][1:][::-1])(tensor)
        with torch.inference_mode():
            pred = self.model(tensor)
        return pred.cpu().numpy()

    def infer_temporal_pytorch(self, img):
        """Runs the per-frame base net, stores its output, and refines from the last seq_len frames."""
        _, refined = self.infer_temporal_pair_pytorch(img)
        return refined

    def infer_temporal_base_pytorch(self, img):
        """Runs only the per-frame base net of a temporal model (no RNN refinement)."""
        tensor = to_scaled_tensor(img).unsqueeze(0).to(self.device)
        tensor = transforms.Resize(self.config["input_shape"][1:][::-1])(tensor)
        with torch.inference_mode():
            base_out = self.model.base_forward(tensor)
        return base_out.cpu().numpy()

    def infer_temporal_pair_pytorch(self, img):
        """Like infer_temporal_pytorch but returns both the per-frame base output
        and the temporally-refined output as (base_pred, refined_pred) numpy arrays."""
        tensor = to_scaled_tensor(img).unsqueeze(0).to(self.device)
        tensor = transforms.Resize(self.config["input_shape"][1:][::-1])(tensor)
        with torch.inference_mode():
            base_out = self.model.base_forward(tensor)
        self.path_buffer.append(base_out)
        seq = list(self.path_buffer)
        while len(seq) < self.seq_len:  # left-pad with the oldest frame until full
            seq.insert(0, seq[0])
        base_seq = torch.stack(seq, dim=1)  # (1, T, ...)
        with torch.inference_mode():
            refined = self.model.refine(base_seq)
        return base_out.cpu().numpy(), refined.cpu().numpy()

    def infer_model_tensorrt(self, img):
        tensor = transforms.Compose(
            [
                to_scaled_tensor,
                transforms.Resize(self.config["input_shape"][1:][::-1]),
            ]
        )(img).contiguous()
        tensor = tensor.numpy()  # convert to numpy
        self.cuda.memcpy_htod(self.bindings[0], tensor)  # copy input to GPU
        self.exectx.execute_v2(self.bindings)  # infer model
        pred = np.empty(self.shapes[1], dtype=np.float32)  # allocate output
        self.cuda.memcpy_dtoh(pred, self.bindings[1])  # copy output to CPU
        return pred

    def split_hypotheses(self, pred):
        """(1, K*D + K) -> list of K (1, D) path vectors, and the K scores."""
        k = self.n_hypotheses
        scores = pred[0, -k:]
        paths = pred[0, :-k].reshape(k, -1)
        return [paths[i][None, :] for i in range(k)], scores

    def detect_hypotheses(self, img):
        """All K ego-path hypotheses for an image, plus their scores.

        Used to separate "did the model propose the right path at all" (oracle)
        from "did it pick the right one" (selection), which is the decomposition
        that says whether a temporal selector is worth building.
        """
        if self.n_hypotheses <= 1:
            return [self.detect(img)], np.ones(1, dtype=np.float32)
        original_shape = img.size
        crop_coords = self.get_crop_coords()
        cropped = img
        if crop_coords is not None:
            xleft, ytop, xright, ybottom = crop_coords
            cropped = img.crop((xleft, ytop, xright + 1, ybottom + 1))
        pred = self.infer_model_pytorch(cropped)
        paths, scores = self.split_hypotheses(pred)
        results = [self.pred_to_result(p, crop_coords, original_shape) for p in paths]
        if isinstance(self.crop_coords, Autocropper):
            self.crop_coords.update(original_shape, results[int(np.argmax(scores))])
        return results, scores

    def detect(self, img):
        """Detects the train ego-path on an image using the model.

        Args:
            img (PIL.Image.Image): Input image on which detection is to be performed.

        Returns:
            list or PIL.Image.Image: Train ego-path detection result, whose type depends on the method used:
                - Classification/Regression: List containing the left and right rails lists of rails point coordinates (x, y).
                - Segmentation: PIL.Image.Image representing the binary mask of detected region.
        """     
        original_shape = img.size
        crop_coords = self.get_crop_coords()
        if crop_coords is not None:
            xleft, ytop, xright, ybottom = crop_coords
            img = img.crop((xleft, ytop, xright + 1, ybottom + 1))

        if self.runtime == "pytorch":
            if not self.temporal:
                pred = self.infer_model_pytorch(img)
            elif self.smoothing_mode in (None, "rnn"):
                pred = self.infer_temporal_pytorch(img)
            else:  # "none", "boxcar" and "ema" operate on the unrefined base output
                pred = self.infer_temporal_base_pytorch(img)
        elif self.runtime == "tensorrt":
            pred = self.infer_model_tensorrt(img)

        if self.n_hypotheses > 1:
            paths, scores = self.split_hypotheses(pred)
            pred = paths[int(np.argmax(scores))]  # score-selected hypothesis
            self.last_scores = scores

        pred = self.apply_smoothing(pred, crop_coords)

        res = self.pred_to_result(pred, crop_coords, original_shape)

        if isinstance(self.crop_coords, Autocropper):
            self.crop_coords.update(original_shape, res)

        return res

    def pred_to_result(self, pred, crop_coords, original_shape):
        """Decode a raw model prediction into the method-specific ego-path result."""
        if self.config["method"] == "classification":
            clf = pred.reshape(2, self.config["anchors"], self.config["classes"] + 1)
            clf = np.argmax(clf, axis=2)
            rails = classifications_to_rails(clf, self.config["classes"])
            rails = scale_rails(rails, crop_coords, original_shape)
            rails = np.round(rails).astype(int)
            return rails.tolist()
        elif self.config["method"] == "regression":
            traj = pred[:, :-1].reshape(2, self.config["anchors"])
            ylim = 1 / (1 + np.exp(-pred[:, -1].item()))  # sigmoid
            rails = regression_to_rails(traj, ylim)
            rails = scale_rails(rails, crop_coords, original_shape)
            rails = np.round(rails).astype(int)
            return rails.tolist()
        elif self.config["method"] == "segmentation":
            mask = pred.squeeze(0).squeeze(0)
            mask = (mask > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask)
            return scale_mask(mask, crop_coords, original_shape)

    def detect_pair(self, img):
        """Detect the ego-path with both the per-frame base net and the temporal RNN.

        Only valid for temporal (RNN) PyTorch models: returns
        (single_frame_result, rnn_result), where the single-frame result is the
        embedded base network's per-frame prediction (no temporal context) and
        the RNN result is the temporally-refined prediction. Computed in a single
        forward pass so the base features are shared. For a non-temporal model,
        returns (detect(img), None).

        This method is deliberately unaffected by the `smoothing` setting: it always
        compares the raw base prediction against the RNN-refined one.
        """
        if not self.temporal or self.runtime != "pytorch":
            return self.detect(img), None

        original_shape = img.size
        crop_coords = self.get_crop_coords()
        cropped = img
        if crop_coords is not None:
            xleft, ytop, xright, ybottom = crop_coords
            cropped = img.crop((xleft, ytop, xright + 1, ybottom + 1))

        base_pred, refined_pred = self.infer_temporal_pair_pytorch(cropped)
        single_res = self.pred_to_result(base_pred, crop_coords, original_shape)
        rnn_res = self.pred_to_result(refined_pred, crop_coords, original_shape)

        if isinstance(self.crop_coords, Autocropper):
            self.crop_coords.update(original_shape, rnn_res)

        return single_res, rnn_res
