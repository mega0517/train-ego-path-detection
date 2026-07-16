"""GPU-side preprocessing for the temporal (RNN) training path.

When `--gpu-preprocess` is on, dataloader workers only read raw JPEG bytes plus
the per-frame crop geometry and a flip flag (see SequencePathsDataset._getitem_thin).
This module turns that light batch into the `(B, T, 3, 512, 512)` float tensor the
model expects, doing the heavy decode/crop/resize/jitter/flip on the GPU so the
CPU dataloader stops being the bottleneck.

The pixel pipeline mirrors the CPU path step-for-step:
  crop -> to float [0, 1] -> resize to input_shape (bilinear, antialias) ->
  per-frame ColorJitter (independent random params) -> optional left-right flip.
"""

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2 as transforms


def _decode_jpegs(jpegs, device):
    """Decodes a list of raw JPEG byte tensors to a list of (3, H, W) uint8 tensors.

    Tries the batched nvJPEG GPU decode first; any image nvJPEG rejects
    (e.g. progressive or CMYK JPEGs) falls back to a CPU decode then moves to the
    device, so the pipeline never silently drops a frame.
    """
    try:
        return torchvision.io.decode_jpeg(jpegs, device=device)
    except Exception:
        imgs = []
        for data in jpegs:
            try:
                imgs.append(torchvision.io.decode_jpeg(data, device=device))
            except Exception:
                cpu_img = torchvision.io.decode_jpeg(data.cpu())
                imgs.append(cpu_img.to(device))
        return imgs


class GpuPreprocess:
    """Callable that converts a thin collated batch into model-ready GPU tensors."""

    def __init__(self, config, device):
        self.device = device
        # (height, width) target, matching to_tensor's Resize(input_shape[1:][::-1]).
        c, h, w = config["input_shape"]
        self.out_size = (h, w)
        self.jitter = transforms.ColorJitter(
            config["brightness"],
            config["contrast"],
            config["saturation"],
            config["hue"],
        )

    def __call__(self, batch):
        """Args: (jpegs, boxes (B,T,4), flip (B,), occl (B,), traj (B,2,A), ylim (B,)).

        Returns: seq (B, T, 3, H, W) float in [0, 1] on device, and the targets
        moved to device: (traj (B,2,A), ylim (B,)).
        """
        jpegs, boxes, flip, occl, traj, ylim = batch
        imgs = _decode_jpegs(jpegs, self.device)  # list of (3, H, W) uint8 on device

        boxes = boxes.to(self.device)
        flip = flip.to(self.device)
        B, T = boxes.shape[0], boxes.shape[1]

        frames = []  # collect (3, out_h, out_w) per (b, t), in row-major (b, t) order
        for b in range(B):
            img = imgs[b]
            for t in range(T):
                left, top, right, height = (int(v) for v in boxes[b, t].tolist())
                # crop bottom is the image bottom (height); right is inclusive in the
                # CPU path (img.crop((left, top, right + 1, height))), so right + 1 here.
                crop = img[:, top:height, left : right + 1]
                crop = crop.to(torch.float32) / 255.0
                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=self.out_size,
                    mode="bilinear",
                    antialias=True,
                ).squeeze(0)
                frames.append(crop)

        seq = torch.stack(frames, dim=0)  # (B*T, 3, H, W)
        # per-frame ColorJitter with independent random params, as in the CPU path.
        seq = self.jitter(seq)
        seq = seq.view(B, T, 3, self.out_size[0], self.out_size[1])

        # left-right flip for the sequences whose flag is set (applied to the crops,
        # matching the CPU path that mirrors each cropped frame).
        flip_mask = flip.bool().view(B, 1, 1, 1, 1)
        seq = torch.where(flip_mask, torch.flip(seq, dims=[-1]), seq)

        # bottom-band occlusion of the LAST frame only (target untouched), matching
        # the CPU path in SequencePathsDataset.__getitem__.
        occl = occl.to(self.device)
        if torch.any(occl > 0):
            from .dataset import SequencePathsDataset

            h = self.out_size[0]
            for b in torch.nonzero(occl > 0).flatten().tolist():
                start = int(round(h * (1 - float(occl[b]))))
                seq[b, -1, :, start:, :] = SequencePathsDataset.OCCLUSION_FILL

        return seq, (traj.to(self.device), ylim.to(self.device))
