"""Prescription & verification for the identity-refiner diagnosis.

Retrain the (near-identity) regression RNN refiner of chromatic-laughter-5RNN on
REAL consecutive OSDaR23 sequences with a temporal-denoising + consistency loss,
then verify on held-out sequences (jitter / consistency IoU), on real labeled
sequences (GT IoU), and on RailSem19 (per-frame IoU unchanged).
"""
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
SEQ_MIN = 40
T = 5           # seq_len
SIGMA_K = 1.0   # injected noise = k * std of consecutive base-output diffs
LAMBDA = 0.3    # consistency weight
EPOCHS = 300
LR = 1e-3
torch.manual_seed(42); np.random.seed(42)

det = Detector(SRC, crop_coords=None, runtime="pytorch", device=device)
model = det.model
resize = transforms.Resize(det.config["input_shape"][1:][::-1])


def base_outputs(folder):
    """Run the base net over a sequence folder -> (N, D) float32 + image size."""
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


# ---------------- A. precompute base outputs ----------------
long_seqs = sorted(
    d for d in glob.glob("/data3/bhkim/datasets/OSDaR23_unzip/*/rgb_center")
    if len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))]) >= SEQ_MIN)
gt_seqs = sorted(
    d for d in glob.glob("/data3/bhkim/datasets/OSDaR23_unzip/*/rgb_center")
    if os.path.exists(os.path.join(d, "egopath_labels.json")))
held_idx = {1, 4, 7, 10}
train_dirs = [d for i, d in enumerate(long_seqs) if i not in held_idx]
held_dirs = [d for i, d in enumerate(long_seqs) if i in held_idx]
print(f"train seqs={len(train_dirs)}, held-out={len(held_dirs)}, GT seqs={len(gt_seqs)}", flush=True)
print("held-out:", [d.split('/')[-2] for d in held_dirs], flush=True)

cache = {}
for d in long_seqs + gt_seqs:
    if d not in cache:
        cache[d] = base_outputs(d)
        print(f"  base outputs: {d.split('/')[-2]} {cache[d][0].shape}", flush=True)

train_arrays = [cache[d][0] for d in train_dirs]
sig = np.concatenate([np.diff(a, axis=0).ravel() for a in train_arrays]).std()
SIGMA = SIGMA_K * float(sig)
print(f"consecutive-diff std={sig:.5f} -> noise sigma={SIGMA:.5f}", flush=True)


def windows(arr):
    """(N, D) -> (N, T, D) left-padded sliding windows (matches inference padding)."""
    N, D = arr.shape
    w = np.empty((N, T, D), dtype=np.float32)
    for t in range(N):
        seq = arr[max(0, t - T + 1):t + 1]
        pad = np.repeat(seq[:1], T - len(seq), axis=0)
        w[t] = np.concatenate([pad, seq], axis=0)
    return w


# ---------------- B. train the refiner ----------------
refiner = model.refiner
before = copy.deepcopy(refiner.state_dict())
opt = torch.optim.Adam(refiner.parameters(), lr=LR)
train_t = [torch.from_numpy(a).to(device) for a in train_arrays]
refiner.train()
for ep in range(EPOCHS):
    tot = td = tc = 0.0
    for arr in train_t:
        W = torch.from_numpy(windows(arr.cpu().numpy())).to(device)   # (N,T,D)
        noise = torch.randn_like(W) * SIGMA
        if torch.rand(1).item() < 0.2:
            noise = noise * 0  # keep some clean windows so identity stays valid too
        refined = refiner(W + noise)                                   # (N,D)
        l_den = torch.mean((refined - arr) ** 2)
        l_con = torch.mean((refined[1:] - refined[:-1]) ** 2)
        loss = l_den + LAMBDA * l_con
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); td += l_den.item(); tc += l_con.item()
    if ep % 50 == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep:3d}  loss={tot:.6f} denoise={td:.6f} consist={tc:.6f}", flush=True)
refiner.eval()

# identity deviation after training
with torch.inference_mode():
    a = train_t[0]
    W = torch.from_numpy(windows(a.cpu().numpy())).to(device)
    dev = (refiner(W) - a).abs()
print(f"refiner deviation from base after training: max={dev.max():.4f} mean={dev.mean():.4f} "
      f"(base scale ~{a.abs().max():.3f})", flush=True)

# ---------------- C. save as a new model dir ----------------
os.makedirs(DST, exist_ok=True)
torch.save(model.state_dict(), os.path.join(DST, "best.pt"))
import shutil
shutil.copy2(os.path.join(SRC, "config.yaml"), os.path.join(DST, "config.yaml"))
print(f"saved {DST}", flush=True)

# ---------------- D. verification ----------------
GRID_N = 48


def decode(vec, size):
    return det.pred_to_result(vec[None, :], None, size)  # [left, right] point lists


def rail_xy(res):
    L, R = res
    if len(L) < 2 or len(R) < 2:
        return None
    la = sorted(L, key=lambda p: p[1]); ra = sorted(R, key=lambda p: p[1])
    ytop = max(la[0][1], ra[0][1]); ybot = min(la[-1][1], ra[-1][1])
    if ybot - ytop < 2:
        return None
    ys = np.linspace(ytop, ybot, GRID_N)
    xl = np.interp(ys, [p[1] for p in la], [p[0] for p in la])
    xr = np.interp(ys, [p[1] for p in ra], [p[0] for p in ra])
    return ys, xl, xr


def iou_masks(a, b):
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return inter / union if union else np.nan


def refined_series(arr, ref):
    W = torch.from_numpy(windows(arr)).to(device)
    with torch.inference_mode():
        return ref(W).cpu().numpy()


def temporal_metrics(preds, size):
    """preds: (N, D) decoded per frame -> (jitter px, consistency IoU)."""
    jit, con = [], []
    prev = None
    for v in preds:
        res = decode(v, size)
        cur = rail_xy(res)
        m = np.array(rails_to_mask(res, size).convert("L")) > 0 if cur else None
        if cur is not None and prev is not None and prev[0] is not None:
            (ys, xl, xr), pm = cur, prev[1]
            pys, pxl, pxr = prev[0]
            y0, y1 = max(ys.min(), pys.min()), min(ys.max(), pys.max())
            if y1 - y0 > 2:
                g = np.linspace(y0, y1, GRID_N)
                d = np.concatenate([np.interp(g, ys, xl) - np.interp(g, pys, pxl),
                                    np.interp(g, ys, xr) - np.interp(g, pys, pxr)])
                jit.append(np.mean(np.abs(d)))
            if m is not None and pm is not None:
                con.append(iou_masks(m, pm))
        prev = (cur, m)
    return np.mean(jit), np.nanmean(con)


# old refiner for comparison
old_refiner = copy.deepcopy(refiner)
old_refiner.load_state_dict(before); old_refiner.eval()

print("\n===== held-out sequences: temporal stability =====", flush=True)
rows = {"single": [], "rnn-old": [], "rnn-new": []}
for d in held_dirs:
    arr, size, _ = cache[d]
    variants = {
        "single": arr,
        "rnn-old": refined_series(arr, old_refiner),
        "rnn-new": refined_series(arr, refiner),
    }
    for k, preds in variants.items():
        rows[k].append(temporal_metrics(preds, size))
for k in ("single", "rnn-old", "rnn-new"):
    j = np.mean([r[0] for r in rows[k]]); c = np.nanmean([r[1] for r in rows[k]])
    print(f"  {k:8s}: jitter={j:.2f} px  consistIoU={c:.4f}", flush=True)

print("\n===== real labeled sequences (3 x 10 frames): GT IoU =====", flush=True)
gt_rows = {"single": [], "rnn-old": [], "rnn-new": []}
for d in gt_seqs:
    arr, size, frames = cache[d]
    labels = json.load(open(os.path.join(d, "egopath_labels.json")))
    variants = {
        "single": arr,
        "rnn-old": refined_series(arr, old_refiner),
        "rnn-new": refined_series(arr, refiner),
    }
    for k, preds in variants.items():
        for v, fn in zip(preds, frames):
            if fn not in labels:
                continue
            gt = labels[fn]
            gt_mask = np.array(rails_to_mask([gt["left_rail"], gt["right_rail"]], size).convert("L")) > 0
            pr_mask = np.array(rails_to_mask(decode(v, size), size).convert("L")) > 0
            gt_rows[k].append(iou_masks(pr_mask, gt_mask))
for k in ("single", "rnn-old", "rnn-new"):
    print(f"  {k:8s}: GT IoU={np.nanmean(gt_rows[k]):.4f}  (n={len(gt_rows[k])})", flush=True)

print("\n===== RailSem19 per-frame IoU (regression check) =====", flush=True)
import random, yaml
from src.utils.common import set_seeds, split_dataset
from src.utils.dataset import PathsDataset
from src.utils.evaluate import IoUEvaluator
with open("configs/global.yaml") as f:
    cfg = yaml.safe_load(f)
set_seeds(cfg["seed"])
with open(cfg["annotations_path"]) as jf:
    idxs = list(range(len(json.load(jf).keys())))
random.shuffle(idxs)
_, _, test_idx = split_dataset(idxs, (cfg["train_prop"], cfg["val_prop"], cfg["test_prop"]))
ds = PathsDataset(cfg["images_path"], cfg["annotations_path"], test_idx, cfg, "segmentation")
for name, path in (("rnn-old", SRC), ("rnn-new", DST)):
    ev = IoUEvaluator(dataset=ds, model_path=path, runtime="pytorch", device=device)
    ev.detector.config["test_iterations"] = 1
    print(f"  {name}: rs19 IoU={ev.evaluate():.4f}", flush=True)
print("\nALL DONE", flush=True)
