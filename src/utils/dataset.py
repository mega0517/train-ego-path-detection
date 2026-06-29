import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from .common import to_scaled_tensor
from .postprocessing import regression_to_rails


class PathsDataset(Dataset):
    def __init__(
        self,
        imgs_path,
        annotations_path,
        indices,
        config,
        method,
        img_aug=False,
        to_tensor=False,
    ):
        """Initializes the dataset for ego-path detection.

        Args:
            imgs_path (str): Path to the images directory.
            annotations_path (str):  Path to the annotations file.
            indices (list): List of indices to use in the dataset.
            config (dict): Data generation configuration.
            method (str): Method to use for ground truth generation ("classification", "regression" or "segmentation").
            img_aug (bool, optional): Whether to use stochastic image adjustment (brightness, contrast, saturation and hue). Defaults to False.
            to_tensor (bool, optional): Whether to return a ready to infer tensor (scaled and possibly resized). Defaults to False.
        """
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.imgs_path = imgs_path
        with open(annotations_path) as json_file:
            self.annotations = json.load(json_file)
        self.imgs = [sorted(self.annotations.keys())[i] for i in indices]
        self.config = config
        self.method = method

        self.img_aug = (
            transforms.ColorJitter(
                self.config["brightness"],
                self.config["contrast"],
                self.config["saturation"],
                self.config["hue"],
            )
            if img_aug
            else None
        )

        self.to_tensor = (
            transforms.Compose(
                [
                    to_scaled_tensor,
                    transforms.Resize(self.config["input_shape"][1:][::-1]),
                ]
            )
            if to_tensor
            else None
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.imgs_path, img_name)).convert("RGB")
        annotation = self.annotations[img_name]
        rails_mask = self.generate_rails_mask(img.size, annotation)
        img, rails_mask = self.random_crop(img, rails_mask)
        img, rails_mask = self.random_flip_lr(img, rails_mask)
        if self.to_tensor:
            img = self.to_tensor(img)
        if self.img_aug:
            img = self.img_aug(img)
        if self.method == "regression":
            path_gt, ylim_gt = self.generate_target_regression(rails_mask)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
                ylim_gt = torch.tensor(ylim_gt)
            return img, path_gt, ylim_gt
        elif self.method == "classification":
            path_gt = self.generate_target_classification(rails_mask)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
            return img, path_gt
        elif self.method == "segmentation":
            segmentation = self.generate_target_segmentation(rails_mask)
            if self.to_tensor:
                segmentation = segmentation.resize(
                    self.config["input_shape"][1:][::-1], Image.NEAREST
                )
                segmentation = to_scaled_tensor(segmentation)
            return img, segmentation

    def generate_rails_mask(self, shape, annotation):
        rails_mask = Image.new("L", shape, 0)
        draw = ImageDraw.Draw(rails_mask)
        rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]
        for rail in rails:
            draw.line([tuple(xy) for xy in rail], fill=1, width=1)
        rails_mask = np.array(rails_mask)
        rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0
        for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
            rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0
        return rails_mask

    def random_crop(self, img, rails_mask):
        left, top, right = self.random_crop_box(img, rails_mask)
        img = img.crop((left, top, right + 1, img.height))
        rails_mask = rails_mask[top:, left : right + 1]
        return img, rails_mask

    def apply_crop_box(self, img, rails_mask, box):
        """Crops an image and its rails mask with a precomputed (left, top, right) box."""
        left, top, right = box
        cropped_img = img.crop((left, top, right + 1, img.height))
        cropped_mask = rails_mask[top:, left : right + 1]
        return cropped_img, cropped_mask

    def random_crop_box(self, img, rails_mask):
        """Computes the random (left, top, right) crop box (bottom is the image bottom).

        Holds the cropping logic previously inlined in random_crop, so the same box
        can be reused to synthesize temporal sequences from a single annotated frame.
        """
        # extract sides rails coordinates (red rectangle in paper fig. 4)
        rails_mask_last_line = rails_mask[-1, :]
        rails_mask_last_line_idx = np.nonzero(rails_mask_last_line)[0]
        most_left_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][0]
        most_right_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][-1]
        # center the crop around the rails (orange rectangle in paper fig. 4)
        base_margin_left = rails_mask_last_line_idx[0] - most_left_rail
        base_margin_right = most_right_rail - rails_mask_last_line_idx[-1]
        max_base_margin = max(base_margin_left, base_margin_right, 0)
        mean_crop_left = rails_mask_last_line_idx[0] - max_base_margin
        mean_crop_right = rails_mask_last_line_idx[-1] + max_base_margin
        # add sides margins (green rectangle in paper fig. 4)
        base_width = mean_crop_right - mean_crop_left + 1
        mean_crop_left -= base_width * self.config["crop_margin_sides"]
        mean_crop_right += base_width * self.config["crop_margin_sides"]
        # random left crop
        largest_margin = max(mean_crop_left, most_left_rail - mean_crop_left)
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_left = round(np.random.normal(mean_crop_left, std_dev))
        if random_crop_left > rails_mask_last_line_idx[0]:
            random_crop_left = 2 * rails_mask_last_line_idx[0] - random_crop_left
        random_crop_left = max(random_crop_left, 0)
        # random right crop
        largest_margin = max(
            mean_crop_right - most_right_rail, img.width - 1 - mean_crop_right
        )
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_right = round(np.random.normal(mean_crop_right, std_dev))
        if random_crop_right < rails_mask_last_line_idx[-1]:
            random_crop_right = 2 * rails_mask_last_line_idx[-1] - random_crop_right
        random_crop_right = min(random_crop_right, img.width - 1)
        # extract top rails coordinates (red rectangle in paper fig. 4)
        most_top_rail = np.nonzero(np.sum(rails_mask, axis=1))[0][0]
        # add top margin (green rectangle in paper fig. 4)
        rail_height = img.height - most_top_rail
        mean_crop_top = most_top_rail - rail_height * self.config["crop_margin_top"]
        # random top crop
        largest_margin = max(mean_crop_top, img.height - 1 - mean_crop_top)
        std_dev = largest_margin * self.config["std_dev_factor_top"]
        random_crop_top = round(np.random.normal(mean_crop_top, std_dev))
        random_crop_top = max(random_crop_top, 0)
        random_crop_top = min(random_crop_top, img.height - 2)  # at least 2 rows
        return random_crop_left, random_crop_top, random_crop_right

    def resize_mask(self, mask, shape):
        height_factor = (shape[0] - 1) / (mask.shape[0] - 1)
        width_factor = (shape[1] - 1) / (mask.shape[1] - 1)
        resized_mask = np.zeros(shape, dtype=np.uint8)
        for i in range(resized_mask.shape[0]):
            row_mask = mask[round(i / height_factor), :]
            row_idx = np.nonzero(row_mask)[0]
            if len(row_idx) == 2:
                resized_mask[i, np.round(row_idx * width_factor).astype(int)] = 1
        return resized_mask

    def random_flip_lr(self, img, rails_mask):
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
            rails_mask = np.fliplr(rails_mask)
        return img, rails_mask

    def generate_target_regression(self, rails_mask):
        unvalid_rows = np.where(np.sum(rails_mask, axis=1) != 2)[0]
        ylim_target = (
            float(1 - (unvalid_rows[-1] + 1) / rails_mask.shape[0])
            if len(unvalid_rows) > 0
            else 1.0
        )
        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], rails_mask.shape[1])
        )
        traj_target = np.array(
            [np.zeros(self.config["anchors"]), np.ones(self.config["anchors"])],
            dtype=np.float32,
        )
        for i in range(self.config["anchors"]):  # it's possible to vectorize this loop
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            if len(rails_points) != 2:
                break
            rails_points_normalized = rails_points / (rails_mask.shape[1] - 1)
            traj_target[:, i] = rails_points_normalized
        return traj_target, ylim_target

    def generate_target_classification(self, rails_mask):
        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], self.config["classes"])
        )
        target = (
            np.ones((2, self.config["anchors"]), dtype=int) * self.config["classes"]
        )
        for i in range(self.config["anchors"]):
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            if len(rails_points) != 2:
                break
            target[:, i] = rails_points
        return target

    def generate_target_segmentation(self, rails_mask):
        target = np.zeros_like(rails_mask, dtype=np.uint8)
        row_indices, col_indices = np.nonzero(rails_mask)
        range_rows = np.arange(row_indices.min(), row_indices.max() + 1)
        for row in reversed(range_rows):
            rails_points = col_indices[row_indices == row]
            if len(rails_points) != 2:
                break
            target[row, rails_points[0] : rails_points[1] + 1] = 255
        return Image.fromarray(target)

    def compute_regression_target(self, idx):
        """Returns the (traj, ylim) regression target for an index without rendering
        the image. get_perspective_weight_limit only needs this crop geometry, so we
        skip the costly pixel decode / resize / augmentation done in __getitem__
        (≈10x the work per sample on the temporal dataset, which would otherwise
        synthesize and tensorize a whole frame sequence just to be thrown away)."""
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.imgs_path, img_name))  # lazy: only .size used
        annotation = self.annotations[img_name]
        rails_mask = self.generate_rails_mask(img.size, annotation)
        left, top, right = self.random_crop_box(img, rails_mask)
        rails_mask = rails_mask[top:, left : right + 1]
        if np.random.rand() < 0.5:  # match the lr-flip augmentation in __getitem__
            rails_mask = np.fliplr(rails_mask)
        return self.generate_target_regression(rails_mask)

    def get_perspective_weight_limit(self, percentile, logger):
        logger.info("\nCalculating perspective weight limit...")
        weights = []
        for i in range(len(self)):
            traj, ylim = self.compute_regression_target(i)
            rails = regression_to_rails(traj, ylim)
            left_rail, right_rail = rails
            rail_width = right_rail[:, 0] - left_rail[:, 0]
            weight = 1 / rail_width
            weights += weight.tolist()
        limit = np.percentile(sorted(weights), percentile)
        logger.info(f"Perspective weight limit: {limit:.2f}")
        return limit


class SequencePathsDataset(PathsDataset):
    """Dataset that synthesizes temporal sequences from the single-frame annotations.

    For each annotated image, a pseudo-sequence of `seq_len` frames is generated by
    starting from a perturbed crop box and smoothly interpolating it toward the
    "current" (last) frame's crop box, mimicking camera motion. The ground truth is
    derived only from the last frame, whose box matches the standard random crop, so
    the existing per-frame targets and losses are reused unchanged.

    NOTE: This is an approximation of real video dynamics (no genuine inter-frame
    motion is available in the single-frame dataset); it provides temporal
    consistency for the RNN to learn to track the path over the frames in memory.
    """

    def __init__(self, *args, seq_len=10, seq_jitter=0.05, **kwargs):
        super(SequencePathsDataset, self).__init__(*args, **kwargs)
        if self.to_tensor is None:
            raise ValueError("SequencePathsDataset requires to_tensor=True")
        self.seq_len = seq_len
        self.seq_jitter = seq_jitter
        # When on, workers only read raw JPEG bytes + crop geometry; the pixel
        # pipeline (decode/resize/jitter/flip) runs on the GPU (see gpu_transforms).
        self.gpu_preprocess = bool(self.config.get("gpu_preprocess", False))

    def generate_sequence_boxes(self, current_box, img_w, img_h):
        """Builds `seq_len` (left, top, right) boxes interpolating to the current box."""
        cl, ct, cr = current_box
        width = max(cr - cl, 1)
        start_left = cl + np.random.normal(0, self.seq_jitter * width)
        start_right = cr + np.random.normal(0, self.seq_jitter * width)
        start_top = ct + np.random.normal(0, self.seq_jitter * (img_h - ct + 1))
        boxes = []
        for i in range(self.seq_len):
            a = i / (self.seq_len - 1) if self.seq_len > 1 else 1.0
            left = int(round(start_left + (cl - start_left) * a))
            right = int(round(start_right + (cr - start_right) * a))
            top = int(round(start_top + (ct - start_top) * a))
            left = max(0, min(left, img_w - 2))
            right = max(left + 1, min(right, img_w - 1))
            top = max(0, min(top, img_h - 2))
            boxes.append((left, top, right))
        boxes[-1] = current_box  # last frame is exactly the current annotated view
        return boxes

    def __getitem__(self, idx):
        if self.gpu_preprocess:
            return self._getitem_thin(idx)

        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.imgs_path, img_name)).convert("RGB")
        annotation = self.annotations[img_name]
        rails_mask = self.generate_rails_mask(img.size, annotation)
        current_box = self.random_crop_box(img, rails_mask)
        boxes = self.generate_sequence_boxes(current_box, img.width, img.height)
        flip = np.random.rand() < 0.5  # consistent flip across the whole sequence

        frames = []
        last_mask = None
        for i, box in enumerate(boxes):
            frame_img, frame_mask = self.apply_crop_box(img, rails_mask, box)
            if flip:
                frame_img = ImageOps.mirror(frame_img)
                frame_mask = np.fliplr(frame_mask)
            frame_tensor = self.to_tensor(frame_img)
            if self.img_aug:
                frame_tensor = self.img_aug(frame_tensor)
            frames.append(frame_tensor)
            if i == len(boxes) - 1:
                last_mask = frame_mask
        seq = torch.stack(frames, dim=0)  # (T, C, H, W)

        if self.method == "regression":
            path_gt, ylim_gt = self.generate_target_regression(last_mask)
            return seq, torch.from_numpy(path_gt), torch.tensor(ylim_gt)
        elif self.method == "classification":
            path_gt = self.generate_target_classification(last_mask)
            return seq, torch.from_numpy(path_gt)
        elif self.method == "segmentation":
            segmentation = self.generate_target_segmentation(last_mask)
            segmentation = segmentation.resize(
                self.config["input_shape"][1:][::-1], Image.NEAREST
            )
            return seq, to_scaled_tensor(segmentation)

    def _getitem_thin(self, idx):
        """GPU-preprocess worker path: no pixel decode, resize or jitter here.

        Returns the raw JPEG bytes plus the crop geometry and flip flag so the
        decode/crop/resize/jitter/flip can run on the GPU (see gpu_transforms).
        The RNG call sites (random_crop_box -> generate_sequence_boxes -> flip)
        are identical to the standard __getitem__ above, so the boxes and flip --
        and therefore the regression target derived from them -- are unchanged.
        Only the per-frame ColorJitter RNG is skipped, which never affected the
        target (it only perturbs pixels, now done on the GPU).
        """
        if self.method != "regression":
            raise NotImplementedError(
                "gpu_preprocess is only implemented for the regression method"
            )
        img_name = self.imgs[idx]
        # lazy open: only header dimensions (.size/.width/.height) are read below,
        # so no JPEG pixels are decoded in the worker.
        img = Image.open(os.path.join(self.imgs_path, img_name))
        annotation = self.annotations[img_name]
        rails_mask = self.generate_rails_mask(img.size, annotation)
        current_box = self.random_crop_box(img, rails_mask)
        boxes = self.generate_sequence_boxes(current_box, img.width, img.height)
        flip = np.random.rand() < 0.5  # consistent flip across the whole sequence

        # per-frame box as (left, top, right, height); the crop bottom is the image
        # bottom, matching apply_crop_box which crops to img.height.
        boxes_t = torch.tensor(
            [(left, top, right, img.height) for (left, top, right) in boxes],
            dtype=torch.long,
        )  # (T, 4)

        # regression target from the last frame's crop (+ flip), exactly as the
        # standard path derives it from last_mask.
        last_left, last_top, last_right = boxes[-1]
        last_mask = rails_mask[last_top:, last_left : last_right + 1]
        if flip:
            last_mask = np.fliplr(last_mask)
        path_gt, ylim_gt = self.generate_target_regression(last_mask)

        jpeg = torchvision.io.read_file(os.path.join(self.imgs_path, img_name))
        return (
            jpeg,
            boxes_t,
            torch.tensor(flip),
            torch.from_numpy(path_gt),
            torch.tensor(ylim_gt),
        )


def gpu_collate_fn(batch):
    """Collate for the gpu_preprocess thin-worker path.

    Each item is (jpeg, boxes (T,4), flip, traj (2,A), ylim). JPEG byte tensors
    have different lengths per image, so they are kept as a list (decoded on the
    GPU); everything else is stacked.
    """
    jpegs = [item[0] for item in batch]
    boxes = torch.stack([item[1] for item in batch], dim=0)  # (B, T, 4)
    flip = torch.stack([item[2] for item in batch], dim=0)  # (B,)
    traj = torch.stack([item[3] for item in batch], dim=0)  # (B, 2, A)
    ylim = torch.stack([item[4] for item in batch], dim=0)  # (B,)
    return jpegs, boxes, flip, traj, ylim
