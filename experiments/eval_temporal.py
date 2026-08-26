"""Temporal-stability evaluation: single-frame vs RNN on continuous sequences.

For each temporal model, detect_pair() yields (single, rnn) predictions from the
SAME forward pass (shared base features) — a controlled comparison. Metrics
(no ground truth needed; camera motion affects both equally):
  - jitter: mean |x_t - x_{t-1}| (px) of each rail sampled on a fixed y-grid,
    over consecutive frames.
  - consistency IoU: IoU(pred_mask_t, pred_mask_{t-1}) of the ego-path region.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
from PIL import Image
from src.utils.interface import Detector
from src.utils.postprocessing import rails_to_mask

device = "cuda"
MODELS = [
    ("brilliant-horse-15RNN", "egopathrnn/weights/brilliant-horse-15RNN"),
    ("chromatic-laughter-5RNN", "egopathrnn/weights/chromatic-laughter-5RNN"),
]
SEQ_MIN = 40  # use continuous sequences with at least this many frames
seq_dirs = sorted(
    d for d in glob.glob("/data3/bhkim/datasets/OSDaR23_unzip/*/rgb_center")
    if len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))]) >= SEQ_MIN
)
print(f"{len(seq_dirs)} sequences (>= {SEQ_MIN} frames)", flush=True)

GRID_N = 48


def boundaries(res, size):
    """(ys, xl, xr) rail boundaries on a fixed y-grid + the path mask, from either
    a segmentation mask or [left, right] point lists."""
    W, H = size
    if isinstance(res, Image.Image):
        arr = np.array(res.convert("L")) > 0
        rows = np.where(arr.any(axis=1))[0]
        if len(rows) < 2:
            return None
        ys = np.linspace(rows.min(), rows.max(), GRID_N)
        xl, xr = [], []
        for y in ys:
            cols = np.where(arr[int(round(y))])[0]
            if len(cols):
                xl.append(cols[0]); xr.append(cols[-1])
            else:
                xl.append(np.nan); xr.append(np.nan)
        return ys, np.array(xl, float), np.array(xr, float), arr
    # point rails -> interpolate x on the grid + build mask for IoU
    L, R = res
    if len(L) < 2 or len(R) < 2:
        return None
    mask = np.array(rails_to_mask([L, R], size).convert("L")) > 0
    ys = None
    la = sorted(L, key=lambda p: p[1]); ra = sorted(R, key=lambda p: p[1])
    ytop = max(la[0][1], ra[0][1]); ybot = min(la[-1][1], ra[-1][1])
    if ybot - ytop < 2:
        return None
    ys = np.linspace(ytop, ybot, GRID_N)
    xl = np.interp(ys, [p[1] for p in la], [p[0] for p in la])
    xr = np.interp(ys, [p[1] for p in ra], [p[0] for p in ra])
    return ys, xl, xr, mask


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else np.nan


for name, path in MODELS:
    det = Detector(path, crop_coords=None, runtime="pytorch", device=device)
    assert det.temporal
    jit = {"single": [], "rnn": []}
    con = {"single": [], "rnn": []}
    nfr = 0
    for sd in seq_dirs:
        frames = sorted(f for f in os.listdir(sd) if f.lower().endswith((".jpg", ".png")))
        det.reset_temporal()
        prev = {"single": None, "rnn": None}
        for fn in frames:
            img = Image.open(os.path.join(sd, fn)); img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            s_res, r_res = det.detect_pair(img)
            nfr += 1
            for key, res in (("single", s_res), ("rnn", r_res)):
                cur = boundaries(res, img.size)
                p = prev[key]
                if cur is not None and p is not None:
                    # jitter on overlapping y-range rows
                    ys, xl, xr, m = cur
                    pys, pxl, pxr, pm = p
                    y0, y1 = max(ys.min(), pys.min()), min(ys.max(), pys.max())
                    if y1 - y0 > 2:
                        g = np.linspace(y0, y1, GRID_N)
                        d_l = np.interp(g, ys, xl) - np.interp(g, pys, pxl)
                        d_r = np.interp(g, ys, xr) - np.interp(g, pys, pxr)
                        d = np.concatenate([d_l, d_r])
                        d = d[~np.isnan(d)]
                        if len(d):
                            jit[key].append(np.mean(np.abs(d)))
                    con[key].append(iou(m, pm))
                prev[key] = cur
        print(f"  [{name}] {os.path.basename(os.path.dirname(sd))}: done", flush=True)
    js, jr = np.mean(jit["single"]), np.mean(jit["rnn"])
    cs, cr = np.nanmean(con["single"]), np.nanmean(con["rnn"])
    print(f"\n== {name} ({det.config['method']}/{det.config['backbone']}), "
          f"{len(seq_dirs)} seqs, {nfr} frames ==", flush=True)
    print(f"  jitter px   : single={js:.2f}  rnn={jr:.2f}  (reduction {100*(js-jr)/js:+.1f}%)", flush=True)
    print(f"  consist. IoU: single={cs:.4f}  rnn={cr:.4f}  (delta {cr-cs:+.4f})", flush=True)
