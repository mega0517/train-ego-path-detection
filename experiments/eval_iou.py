"""IoU-only single-frame vs RNN comparison (mirrors eval.py, no latency cooldown)."""
import json, os, random, sys
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
import yaml
from src.utils.common import set_seeds, split_dataset
from src.utils.dataset import PathsDataset
from src.utils.evaluate import IoUEvaluator

device = "cuda"
with open("configs/global.yaml") as f:
    config = yaml.safe_load(f)
set_seeds(config["seed"])
with open(config["annotations_path"]) as jf:
    indices = list(range(len(json.load(jf).keys())))
random.shuffle(indices)
_, _, test_indices = split_dataset(
    indices, (config["train_prop"], config["val_prop"], config["test_prop"]))
test_dataset = PathsDataset(
    config["images_path"], config["annotations_path"], test_indices, config, "segmentation")
print(f"test set: {len(test_dataset)} images", flush=True)

roots = ["egopath/weights", "egopathrnn/weights"]
models = []
for root in roots:
    if not os.path.isdir(root):
        continue
    for name in sorted(os.listdir(root)):
        mp = os.path.join(root, name)
        if os.path.isdir(mp) and os.path.exists(os.path.join(mp, "config.yaml")):
            models.append((name, mp, "rnn" if "egopathrnn" in root or name.endswith("RNN") else "single"))

rows = []
for name, mp, mtype in models:
    try:
        ev = IoUEvaluator(dataset=test_dataset, model_path=mp, runtime="pytorch", device=device)
        ev.detector.config["test_iterations"] = 1  # single pass for speed
        method = ev.detector.config.get("method", "?")
        backbone = ev.detector.config.get("backbone", "?")
        iou = ev.evaluate()
        rows.append((name, mtype, method, backbone, iou))
        print(f"[done] {mtype:6s} {name:24s} {method}/{backbone}  IoU={iou:.4f}", flush=True)
    except Exception as e:
        print(f"[skip] {name} ({mtype}): {type(e).__name__}: {e}", flush=True)

print("\n===== IoU comparison (single vs RNN) =====", flush=True)
rows.sort(key=lambda r: (r[0].replace("RNN", ""), r[1]))
print(f"{'model':26s} {'type':7s} {'method/backbone':28s} {'IoU':>7s}")
for name, mtype, method, backbone, iou in rows:
    print(f"{name:26s} {mtype:7s} {method+'/'+backbone:28s} {iou:7.4f}")
# base vs its RNN delta
by = {n: iou for n, t, m, b, iou in rows}
print("\n----- base → RNN (IoU delta) -----")
for name, mtype, method, backbone, iou in rows:
    if mtype == "single" and (name + "RNN") in by:
        d = by[name + "RNN"] - iou
        print(f"{name:26s} single={iou:.4f}  RNN={by[name+'RNN']:.4f}  Δ={d:+.4f}")
