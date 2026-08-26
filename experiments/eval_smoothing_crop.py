"""Does boxcar smoothing still help when the Autocropper is live?

The raw prediction vector is expressed in coordinates relative to the current crop
window, but Detector.apply_smoothing averages raw predictions *before* pred_to_result
maps them back to absolute image coordinates. With crop_coords="auto" the Autocropper
moves the window every frame, so the buffered predictions live in different frames of
reference. eval_ema.py measured smoothing with crop_coords=None only; detect.py defaults
to --crop auto. This script measures GT IoU through the full Detector.detect() path
(Autocropper live) for both crop settings.

A fresh Detector is built per (configuration, sequence) so neither the Autocropper nor
the smoothing buffer leaks across sequences.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
from PIL import Image
from src.utils.interface import Detector
from src.utils.postprocessing import rails_to_mask

MODEL = "egopathrnn/weights/chromatic-laughter-5RNN"
DEVICE = "cuda"

gt_dirs = sorted(
    d for d in glob.glob("/data3/bhkim/datasets/OSDaR23_unzip/*/rgb_center")
    if os.path.exists(os.path.join(d, "egopath_labels.json")))

CONFIGS = [
    ("crop=None  smoothing=none   ", None, "none"),
    ("crop=None  smoothing=boxcar5", None, "boxcar5"),
    ("crop=auto  smoothing=none   ", "auto", "none"),
    ("crop=auto  smoothing=boxcar5", "auto", "boxcar5"),
]


def load(path):
    img = Image.open(path); img.load()
    return img.convert("RGB") if img.mode != "RGB" else img


def iou_masks(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else np.nan


def run_sequence(crop, smoothing, folder, labels):
    """Runs the full detect() path over a sequence, returns the per-labeled-frame IoUs."""
    det = Detector(MODEL, crop, "pytorch", DEVICE, smoothing=smoothing)
    det.reset_temporal()
    frames = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
    ious = []
    for fn in frames:
        img = load(os.path.join(folder, fn))
        res = det.detect(img)
        if fn not in labels:
            continue
        gt = labels[fn]
        gtm = np.array(rails_to_mask([gt["left_rail"], gt["right_rail"]], img.size).convert("L")) > 0
        prm = np.array(rails_to_mask(res, img.size).convert("L")) > 0
        ious.append(iou_masks(prm, gtm))
    return ious


results = {name: {} for name, _, _ in CONFIGS}
labels_cache = {d: json.load(open(os.path.join(d, "egopath_labels.json"))) for d in gt_dirs}
for name, crop, smoothing in CONFIGS:
    for d in gt_dirs:
        results[name][d] = run_sequence(crop, smoothing, d, labels_cache[d])
    print(f"done: {name}", flush=True)

print("\n### GT IoU through the full Detector.detect() path ###")
head = "sequence".ljust(34) + "".join(n.strip().ljust(30) for n, _, _ in CONFIGS)
print(head)
for d in gt_dirs:
    row = os.path.basename(os.path.dirname(d))[:32].ljust(34)
    for name, _, _ in CONFIGS:
        row += f"{np.nanmean(results[name][d]):.4f} (n={len(results[name][d])})".ljust(30)
    print(row)
print("-" * len(head))
row = "AGGREGATE (frame-weighted)".ljust(34)
for name, _, _ in CONFIGS:
    allv = [v for d in gt_dirs for v in results[name][d]]
    row += f"{np.nanmean(allv):.4f} (n={len(allv)})".ljust(30)
print(row)

for crop in ("crop=None", "crop=auto"):
    base = [v for d in gt_dirs for v in results[f"{crop}  smoothing=none   "][d]]
    box = [v for d in gt_dirs for v in results[f"{crop}  smoothing=boxcar5"][d]]
    delta = np.nanmean(box) - np.nanmean(base)
    print(f"\n{crop}: boxcar5 - none = {delta:+.4f} GT IoU"
          + f"  ({'boxcar wins' if delta > 0 else 'boxcar loses'})")
print("\nALL DONE", flush=True)
