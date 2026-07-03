"""Robustness verification: under per-frame estimation perturbation (noise added to
base outputs, same for all variants), does the retrained denoising refiner help?"""
import copy, glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
from PIL import Image
from torchvision.transforms import v2 as transforms
from src.utils.interface import Detector
from src.utils.common import to_scaled_tensor
from src.utils.postprocessing import rails_to_mask

device = "cuda"
SRC = "egopathrnn/weights/chromatic-laughter-5RNN"
DST = "egopathrnn/weights/chromatic-laughter-5RNN-tc"
T, GRID_N = 5, 48
torch.manual_seed(0); np.random.seed(0)

det_old = Detector(SRC, None, "pytorch", device)
det_new = Detector(DST, None, "pytorch", device)
model = det_old.model
resize = transforms.Resize(det_old.config["input_shape"][1:][::-1])
long_seqs = sorted(
    d for d in glob.glob("/data3/bhkim/datasets/OSDaR23_unzip/*/rgb_center")
    if len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))]) >= 40)
held_dirs = [d for i, d in enumerate(long_seqs) if i in {1, 4, 7, 10}]
gt_dirs = sorted(
    d for d in glob.glob("/data3/bhkim/datasets/OSDaR23_unzip/*/rgb_center")
    if os.path.exists(os.path.join(d, "egopath_labels.json")))


def base_outputs(folder):
    frames = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
    outs, size = [], None
    for fn in frames:
        img = Image.open(os.path.join(folder, fn)); img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        size = img.size
        t = resize(to_scaled_tensor(img).unsqueeze(0)).to(device)
        with torch.inference_mode():
            outs.append(model.base_forward(t).cpu().numpy()[0])
    return np.stack(outs).astype(np.float32), size, frames


def windows(arr):
    N, D = arr.shape
    w = np.empty((N, T, D), dtype=np.float32)
    for t in range(N):
        seq = arr[max(0, t - T + 1):t + 1]
        pad = np.repeat(seq[:1], T - len(seq), axis=0)
        w[t] = np.concatenate([pad, seq], axis=0)
    return w


def refined_series(arr, refiner):
    W = torch.from_numpy(windows(arr)).to(device)
    with torch.inference_mode():
        return refiner(W).cpu().numpy()


def decode(vec, size):
    return det_old.pred_to_result(vec[None, :], None, size)


def rail_xy(res):
    L, R = res
    if len(L) < 2 or len(R) < 2:
        return None
    la = sorted(L, key=lambda p: p[1]); ra = sorted(R, key=lambda p: p[1])
    ytop = max(la[0][1], ra[0][1]); ybot = min(la[-1][1], ra[-1][1])
    if ybot - ytop < 2:
        return None
    ys = np.linspace(ytop, ybot, GRID_N)
    return (ys, np.interp(ys, [p[1] for p in la], [p[0] for p in la]),
            np.interp(ys, [p[1] for p in ra], [p[0] for p in ra]))


def iou_masks(a, b):
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return inter / union if union else np.nan


def temporal_metrics(preds, size):
    jit, con, prev = [], [], None
    for v in preds:
        res = decode(v, size)
        cur = rail_xy(res)
        m = np.array(rails_to_mask(res, size).convert("L")) > 0 if cur else None
        if cur is not None and prev is not None and prev[0] is not None:
            ys, xl, xr = cur; pys, pxl, pxr = prev[0]
            y0, y1 = max(ys.min(), pys.min()), min(ys.max(), pys.max())
            if y1 - y0 > 2:
                g = np.linspace(y0, y1, GRID_N)
                d = np.concatenate([np.interp(g, ys, xl) - np.interp(g, pys, pxl),
                                    np.interp(g, ys, xr) - np.interp(g, pys, pxr)])
                jit.append(np.mean(np.abs(d)))
            if m is not None and prev[1] is not None:
                con.append(iou_masks(m, prev[1]))
        prev = (cur, m)
    return np.mean(jit), np.nanmean(con)


cache = {d: base_outputs(d) for d in held_dirs + gt_dirs}
print("base outputs cached", flush=True)

for SIG in (0.004, 0.008):
    print(f"\n##### perturbation sigma={SIG} (base scale ~0.65) #####", flush=True)
    print("--- held-out: temporal stability under perturbation ---", flush=True)
    agg = {"single": [], "rnn-old": [], "rnn-new": []}
    for d in held_dirs:
        arr, size, _ = cache[d]
        noisy = arr + np.random.randn(*arr.shape).astype(np.float32) * SIG
        variants = {
            "single": noisy,
            "rnn-old": refined_series(noisy, det_old.model.refiner),
            "rnn-new": refined_series(noisy, det_new.model.refiner),
        }
        for k, p in variants.items():
            agg[k].append(temporal_metrics(p, size))
    for k in ("single", "rnn-old", "rnn-new"):
        j = np.mean([r[0] for r in agg[k]]); c = np.nanmean([r[1] for r in agg[k]])
        print(f"  {k:8s}: jitter={j:.2f} px  consistIoU={c:.4f}", flush=True)

    print("--- labeled seqs: GT IoU under perturbation ---", flush=True)
    gt_agg = {"single": [], "rnn-old": [], "rnn-new": []}
    for d in gt_dirs:
        arr, size, frames = cache[d]
        labels = json.load(open(os.path.join(d, "egopath_labels.json")))
        noisy = arr + np.random.randn(*arr.shape).astype(np.float32) * SIG
        variants = {
            "single": noisy,
            "rnn-old": refined_series(noisy, det_old.model.refiner),
            "rnn-new": refined_series(noisy, det_new.model.refiner),
        }
        for k, p in variants.items():
            for v, fn in zip(p, frames):
                if fn not in labels:
                    continue
                gt = labels[fn]
                gtm = np.array(rails_to_mask([gt["left_rail"], gt["right_rail"]], size).convert("L")) > 0
                prm = np.array(rails_to_mask(decode(v, size), size).convert("L")) > 0
                gt_agg[k].append(iou_masks(prm, gtm))
    for k in ("single", "rnn-old", "rnn-new"):
        print(f"  {k:8s}: GT IoU={np.nanmean(gt_agg[k]):.4f} (n={len(gt_agg[k])})", flush=True)
print("\nALL DONE", flush=True)
