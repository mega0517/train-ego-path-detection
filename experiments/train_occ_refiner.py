"""Occlusion prescription v2: train the temporal refiner to recover the ego-path
hidden by the train nose (turnout occlusion), directly on cached base outputs.

Why not train.py --seq-occlusion alone: the standard perspective-weighted loss
down-weights the bottom (occluded) anchors ~10-20x and rs19 is dominated by
straight tracks where an occluded frame's extrapolated output is indistinguishable
from a genuinely straight one — the identity refiner stays (near-)optimal and the
run converges back to identity (verified: delta <= 0.005 IoU at all occlusion
levels). Here the base net is frozen, so we can cache its outputs once and train
the refiner with a UNIFORM-weight trajectory loss (the occlusion-recovery target
lives in the bottom anchors) and a high occlusion rate.

Windows mirror inference (interface.infer_temporal_pair_pytorch): 4 clean past
outputs + the current output, where the current frame is occluded with prob
P_OCC at a fraction drawn from FRACS.
"""
import json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
import yaml
from src.utils.common import set_seeds, split_dataset
from src.utils.dataset import SequencePathsDataset
from src.utils.interface import Detector

SRC = "egopathrnn/weights/chromatic-laughter-5RNN"
DST = "egopathrnn/weights/chromatic-laughter-5RNN-occ"
CACHE = "output/occ_refiner_cache.npz"
FRACS = [0.15, 0.25, 0.35, 0.45]
FILL = 15 / 255.0
P_OCC = 0.5
EPOCHS = 300
LR = 1e-3
BATCH = 256
device = "cuda"
torch.manual_seed(0); np.random.seed(0); random.seed(0)

det = Detector(SRC, None, "pytorch", device)
model = det.model
T = det.config["seq_len"]
H_in = det.config["input_shape"][1]

# ---------------- A. cache base outputs (clean 5 + occluded last x4) ----------------
if not os.path.exists(CACHE):
    cfg = dict(det.config); cfg["gpu_preprocess"] = False
    set_seeds(cfg["seed"])
    with open(cfg["annotations_path"]) as jf:
        idxs = list(range(len(json.load(jf).keys())))
    random.shuffle(idxs)
    tr, va, _ = split_dataset(idxs, (cfg["train_prop"], cfg["val_prop"], cfg["test_prop"]))
    set_seeds(cfg["seed"])

    def build(indices, tag):
        ds = SequencePathsDataset(
            cfg["images_path"], cfg["annotations_path"], indices, cfg, "regression",
            img_aug=False, to_tensor=True, seq_len=T, seq_jitter=cfg["seq_jitter"])
        loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=16)
        CL, OC, TR, YL = [], [], [], []
        done = 0
        for seq, traj, ylim in loader:
            seq = seq.to(device)  # (B, T, C, H, W)
            B = seq.shape[0]
            occ_last = []
            for frac in FRACS:
                last = seq[:, -1].clone()
                last[:, :, int(round(H_in * (1 - frac))):, :] = FILL
                occ_last.append(last)
            occ_last = torch.stack(occ_last, dim=1)  # (B, F, C, H, W)
            with torch.no_grad():
                clean = model.base_forward(seq.flatten(0, 1)).view(B, T, -1)
                occ = model.base_forward(occ_last.flatten(0, 1)).view(B, len(FRACS), -1)
            CL.append(clean.cpu().numpy()); OC.append(occ.cpu().numpy())
            TR.append(traj.numpy()); YL.append(ylim.numpy())
            done += B
            if done % 800 < 16:
                print(f"  cache {tag}: {done}/{len(ds)}", flush=True)
        return (np.concatenate(CL), np.concatenate(OC),
                np.concatenate(TR).astype(np.float32), np.concatenate(YL).astype(np.float32))

    tr_cl, tr_oc, tr_tr, tr_yl = build(tr, "train")
    va_cl, va_oc, va_tr, va_yl = build(va, "val")
    os.makedirs("output", exist_ok=True)
    np.savez_compressed(CACHE, tr_cl=tr_cl, tr_oc=tr_oc, tr_tr=tr_tr, tr_yl=tr_yl,
                        va_cl=va_cl, va_oc=va_oc, va_tr=va_tr, va_yl=va_yl)
    print(f"cached -> {CACHE}", flush=True)
z = np.load(CACHE)
tr_cl, tr_oc = torch.from_numpy(z["tr_cl"]).to(device), torch.from_numpy(z["tr_oc"]).to(device)
tr_tr, tr_yl = torch.from_numpy(z["tr_tr"]).to(device), torch.from_numpy(z["tr_yl"]).to(device)
va_cl, va_oc = torch.from_numpy(z["va_cl"]).to(device), torch.from_numpy(z["va_oc"]).to(device)
va_tr, va_yl = torch.from_numpy(z["va_tr"]).to(device), torch.from_numpy(z["va_yl"]).to(device)
N, Nv = tr_cl.shape[0], va_cl.shape[0]
print(f"cache: train {N}, val {Nv}, D={tr_cl.shape[-1]}", flush=True)

smae = nn.SmoothL1Loss(reduction="none", beta=0.005)
ylim_smae = nn.SmoothL1Loss(reduction="mean", beta=0.015)
A = tr_tr.shape[-1]


def uniform_loss(pred, traj_t, ylim_t):
    """Trajectory SmoothL1 masked by ylim with UNIFORM anchor weights + ylim loss."""
    traj_p = pred[:, :-1].view_as(traj_t)
    ylim_p = pred[:, -1]
    se = smae(traj_p, traj_t)  # (B, 2, A)
    rng = torch.arange(A, device=pred.device).expand(traj_t.size(0), -1)
    mask = (rng <= (ylim_t * (A - 1)).unsqueeze(1)).float()  # (B, A)
    tl = (se * mask.unsqueeze(1)).sum(dim=(1, 2)) / (2 * mask.sum(dim=1).clamp(min=1))
    tl[ylim_t == 0] = 0
    return tl.mean() + 0.5 * ylim_smae(torch.sigmoid(ylim_p), ylim_t)


def make_windows(cl, oc, occ_choice):
    """cl (B,T,D), oc (B,F,D), occ_choice (B,) in [-1, F) — -1 keeps clean last."""
    w = cl.clone()
    sel = occ_choice >= 0
    if sel.any():
        w[sel, -1] = oc[sel, occ_choice[sel]]
    return w


refiner = model.refiner
opt = torch.optim.Adam(refiner.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
F = len(FRACS)


def val_losses():
    refiner.eval()
    out = {}
    with torch.no_grad():
        for name, w in (("clean", va_cl),
                        ("occ35", make_windows(va_cl, va_oc, torch.full((Nv,), 2, dtype=torch.long, device=device)))):
            ident = uniform_loss(w[:, -1], va_tr, va_yl).item()
            box = uniform_loss(w.mean(dim=1), va_tr, va_yl).item()
            ref = uniform_loss(refiner(w), va_tr, va_yl).item()
            out[name] = (ident, box, ref)
    refiner.train()
    return out


v = val_losses()
print(f"[before] clean: id={v['clean'][0]:.5f} box={v['clean'][1]:.5f} ref={v['clean'][2]:.5f} | "
      f"occ35: id={v['occ35'][0]:.5f} box={v['occ35'][1]:.5f} ref={v['occ35'][2]:.5f}", flush=True)

best = (1e9, None)
refiner.train()
for ep in range(EPOCHS):
    perm = torch.randperm(N, device=device)
    tot = 0.0
    for s in range(0, N, BATCH):
        bi = perm[s:s + BATCH]
        occ_choice = torch.randint(0, F, (len(bi),), device=device)
        occ_choice[torch.rand(len(bi), device=device) >= P_OCC] = -1
        w = make_windows(tr_cl[bi], tr_oc[bi], occ_choice)
        loss = uniform_loss(refiner(w), tr_tr[bi], tr_yl[bi])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(bi)
    sched.step()
    if ep % 10 == 0 or ep == EPOCHS - 1:
        v = val_losses()
        score = v["clean"][2] + v["occ35"][2]
        star = ""
        if score < best[0]:
            best = (score, {k: t.detach().cpu().clone() for k, t in refiner.state_dict().items()})
            star = " *"
        print(f"ep {ep:3d} train={tot/N:.5f} | val clean ref={v['clean'][2]:.5f} (id {v['clean'][0]:.5f}) | "
              f"val occ35 ref={v['occ35'][2]:.5f} (id {v['occ35'][0]:.5f}, box {v['occ35'][1]:.5f}){star}", flush=True)

refiner.load_state_dict(best[1])
refiner.eval()
v = val_losses()
print(f"[best] clean: id={v['clean'][0]:.5f} box={v['clean'][1]:.5f} ref={v['clean'][2]:.5f} | "
      f"occ35: id={v['occ35'][0]:.5f} box={v['occ35'][1]:.5f} ref={v['occ35'][2]:.5f}", flush=True)

os.makedirs(DST, exist_ok=True)
torch.save(model.state_dict(), os.path.join(DST, "best.pt"))
with open(os.path.join(SRC, "config.yaml")) as f:
    cfg_out = yaml.safe_load(f)
cfg_out.update({"seq_occlusion_prob": P_OCC, "seq_occlusion_min": FRACS[0],
                "seq_occlusion_max": FRACS[-1], "occ_refiner_uniform_loss": True,
                "occ_refiner_epochs": EPOCHS, "occ_refiner_lr": LR})
with open(os.path.join(DST, "config.yaml"), "w") as f:
    yaml.dump(cfg_out, f)
print(f"saved {DST}\nALL DONE", flush=True)
