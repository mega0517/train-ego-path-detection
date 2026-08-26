"""Occlusion diagnosis: does the temporal RNN preserve the ego-path when the
current frame's lower region is occluded (train nose hiding an approaching
turnout)?

Protocol (matches the intended real-world failure): the refiner window holds
CLEAN base outputs for the past frames (the turnout was visible while farther
away) and the base output of the OCCLUDED current frame. GT is the current
frame's label, so any recovery must come from temporal memory.
  - single = base output of the occluded current frame (no temporal context)
  - rnn    = refiner over [clean past .. occluded current]

Part A: OSDaR23 labeled real sequences (station turnout seqs + calibration).
Part B: rs19 test pseudo-sequences (training distribution), occluding the
        last frame of each synthesized sequence.
"""
import glob, json, os, random, sys
import numpy as np
import torch
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
import yaml
from PIL import Image
from torchvision.transforms import v2 as transforms
from src.utils.common import set_seeds, split_dataset, to_scaled_tensor
from src.utils.dataset import SequencePathsDataset
from src.utils.evaluate import compute_iou
from src.utils.interface import Detector
from src.utils.postprocessing import rails_to_mask, regression_to_rails, scale_rails

device = "cuda"
MODELS = [  # refiners sharing the same frozen base (chromatic-laughter-5)
    ("rnn", "egopathrnn/weights/chromatic-laughter-5RNN"),
    ("rnn-tc", "egopathrnn/weights/chromatic-laughter-5RNN-tc"),
]
EXTRA = [m for m in sys.argv[1:] if os.path.isdir(m)]  # e.g. occlusion-trained run dir
MODELS += [(os.path.basename(m), m) for m in EXTRA]
OCCLUSIONS = [0.0, 0.10, 0.20, 0.30, 0.40]  # fraction of image height, from bottom
FILL = (15, 15, 15)  # dark train-nose-like fill
T = 5
RS19_N = 300  # rs19 pseudo-sequence samples per occlusion level
torch.manual_seed(0); np.random.seed(0); random.seed(0)

dets = {name: Detector(path, None, "pytorch", device) for name, path in MODELS}
det0 = dets[MODELS[0][0]]
resize = transforms.Resize(det0.config["input_shape"][1:][::-1])
IN_SIZE = tuple(det0.config["input_shape"][1:][::-1])  # (W, H) = (512, 512)

# verify shared base so we can reuse cached base outputs across refiners
bases = {name: d.config.get("base_model", "?") for name, d in dets.items()}
print(f"models: { {n: (bases[n], dets[n].config['backbone']) for n in dets} }", flush=True)
shared_base = len(set(bases.values())) == 1


def occlude(img, frac):
    if frac <= 0:
        return img
    img = img.copy()
    W, H = img.size
    img.paste(FILL, (0, int(round(H * (1 - frac))), W, H))
    return img


def base_out(det, img):
    t = resize(to_scaled_tensor(img).unsqueeze(0)).to(device)
    with torch.no_grad():
        return det.model.base_forward(t).cpu().numpy()[0]


def refine_window(det, window):  # window: list of (D,) arrays, len T
    seq = torch.from_numpy(np.stack(window)[None]).to(device)  # (1, T, D)
    with torch.no_grad():
        return det.model.refine(seq).cpu().numpy()[0]


def decode_mask(det, vec, size):
    res = det.pred_to_result(vec[None, :], None, size)
    return rails_to_mask(res, size)


# ---------------- Part A: OSDaR23 labeled real sequences ----------------
GT_DIRS = sorted(
    d for d in glob.glob("/data3/bhkim/datasets/OSDaR23_unzip/*/rgb_center")
    if os.path.exists(os.path.join(d, "egopath_labels.json")))
print(f"\n=== Part A: OSDaR23 real sequences ({len(GT_DIRS)} labeled) ===", flush=True)

# cache base outputs: clean (all frames) + occluded (labeled frames) per level
results_A = {}  # (seqname, frac, variant) -> [ious]
for d in GT_DIRS:
    seqname = os.path.basename(os.path.dirname(d))
    labels = json.load(open(os.path.join(d, "egopath_labels.json")))
    frames = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
    imgs = {}
    for fn in frames:
        im = Image.open(os.path.join(d, fn)); im.load()
        imgs[fn] = im.convert("RGB") if im.mode != "RGB" else im
    size = imgs[frames[0]].size
    clean = {name: {fn: base_out(det, imgs[fn]) for fn in frames}
             for name, det in ([next(iter(dets.items()))] if shared_base else dets.items())}

    def clean_of(name, fn):
        return clean[next(iter(clean))][fn] if shared_base else clean[name][fn]

    for frac in OCCLUSIONS:
        occ = {}
        for fn in frames:
            if fn in labels:
                occ[fn] = base_out(det0, occlude(imgs[fn], frac)) if shared_base else None
        for name, det in dets.items():
            for ti, fn in enumerate(frames):
                if fn not in labels:
                    continue
                cur = occ[fn] if shared_base else base_out(det, occlude(imgs[fn], frac))
                past = [clean_of(name, frames[j]) for j in range(max(0, ti - T + 1), ti)]
                window = ([past[0]] * (T - 1 - len(past)) + past + [cur]
                          if past else [cur] * T)
                gt = labels[fn]
                gtm = rails_to_mask([gt["left_rail"], gt["right_rail"]], size)
                for variant, vec in (("single", cur),
                                     ("rnn", refine_window(det, window)),
                                     ("boxcar", np.mean(np.stack(window), axis=0))):
                    iou = compute_iou(decode_mask(det, vec, size), gtm)
                    results_A.setdefault((seqname, frac, f"{name}/{variant}"), []).append(iou)
    print(f"  {seqname}: done ({len([f for f in frames if f in labels])} labeled frames)", flush=True)

# report: station GT seqs (human-corrected) vs calibration (pseudo-GT) separately
groups = {
    "turnout-GT(30f)": [s for s in set(k[0] for k in results_A) if "station" in s],
    "calib-pseudoGT(100f)": [s for s in set(k[0] for k in results_A) if "calibration" in s],
}
for gname, seqs in groups.items():
    print(f"\n--- {gname} ---", flush=True)
    hdr = "occl%  " + "  ".join(f"{n}: single/rnn/boxcar".ljust(28) for n in dets)
    print(hdr, flush=True)
    for frac in OCCLUSIONS:
        row = [f"{int(frac*100):>4d}%"]
        for name in dets:
            s = np.mean([i for sq in seqs for i in results_A.get((sq, frac, f"{name}/single"), [])])
            r = np.mean([i for sq in seqs for i in results_A.get((sq, frac, f"{name}/rnn"), [])])
            b = np.mean([i for sq in seqs for i in results_A.get((sq, frac, f"{name}/boxcar"), [])])
            row.append(f"{s:.4f} / {r:.4f} (Δ{r-s:+.4f}) / {b:.4f}")
        print("  ".join(row), flush=True)

# ---------------- Part B: rs19 pseudo-sequences (training distribution) ----------------
print(f"\n=== Part B: rs19 test pseudo-sequences (N={RS19_N}/level) ===", flush=True)
cfg = dict(det0.config)
cfg["gpu_preprocess"] = False
set_seeds(cfg["seed"])
with open(cfg["annotations_path"]) as jf:
    idxs = list(range(len(json.load(jf).keys())))
random.shuffle(idxs)
_, _, test_idx = split_dataset(idxs, (cfg["train_prop"], cfg["val_prop"], cfg["test_prop"]))
ds = SequencePathsDataset(
    cfg["images_path"], cfg["annotations_path"], test_idx, cfg, cfg["method"],
    img_aug=False, to_tensor=True, seq_len=cfg["seq_len"], seq_jitter=cfg["seq_jitter"])
print(f"test set: {len(ds)} frames", flush=True)

results_B = {}  # (frac, name/variant) -> [ious]
H_in = det0.config["input_shape"][1]
sample_ids = list(range(len(ds)))
random.Random(0).shuffle(sample_ids)
sample_ids = sample_ids[:RS19_N]
for si, i in enumerate(sample_ids):
    seq, traj_gt, ylim_gt = ds[i]  # (T,C,H,W), (2,A), scalar — same seq reused per level
    gt_rails = regression_to_rails(traj_gt.numpy(), float(ylim_gt))
    gt_rails = scale_rails(gt_rails, None, IN_SIZE)
    gtm = rails_to_mask(np.round(gt_rails).astype(int).tolist(), IN_SIZE)
    seq_d = seq.to(device)
    with torch.no_grad():
        base_clean = det0.model.base_forward(seq_d).cpu().numpy()  # (T, D)
    for frac in OCCLUSIONS:
        last = seq_d[-1:].clone()
        if frac > 0:
            last[:, :, int(round(H_in * (1 - frac))):, :] = 15 / 255.0
        with torch.no_grad():
            cur = det0.model.base_forward(last).cpu().numpy()[0]
        window = list(base_clean[:-1]) + [cur]
        for name, det in dets.items():
            for variant, vec in (("single", cur),
                                 ("rnn", refine_window(det, window)),
                                 ("boxcar", np.mean(np.stack(window), axis=0))):
                iou = compute_iou(decode_mask(det, vec, IN_SIZE), gtm)
                results_B.setdefault((frac, f"{name}/{variant}"), []).append(iou)
    if (si + 1) % 100 == 0:
        print(f"  {si+1}/{len(sample_ids)}…", flush=True)

print("\n--- rs19 pseudo-seq ---", flush=True)
for frac in OCCLUSIONS:
    row = [f"{int(frac*100):>4d}%"]
    for name in dets:
        s = np.mean(results_B[(frac, f"{name}/single")])
        r = np.mean(results_B[(frac, f"{name}/rnn")])
        b = np.mean(results_B[(frac, f"{name}/boxcar")])
        row.append(f"{s:.4f} / {r:.4f} (Δ{r-s:+.4f}) / {b:.4f}")
    print("  ".join(row), flush=True)

out = {"A": {f"{k[0]}|{k[1]}|{k[2]}": float(np.mean(v)) for k, v in results_A.items()},
       "B": {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in results_B.items()}}
os.makedirs("output", exist_ok=True)
tag = "_".join(n for n, _ in MODELS[2:]) or "baseline"
with open(f"output/eval_occlusion_{tag}.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nsaved output/eval_occlusion_{tag}.json\nALL DONE", flush=True)
