import json
import os
import random
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import yaml

from src.utils.common import (
    set_seeds,
    split_dataset,
)
from src.utils.dataset import PathsDataset
from src.utils.evaluate import IoUEvaluator, LatencyEvaluator

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

methods = [  # methods to evaluate
    "classification",
    "regression",
    "segmentation",
]
backbones = [  # backbones to evaluate
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "resnet18",
    "resnet34",
    "resnet50",
]
runtimes = [  # runtimes to evaluate ("tensorrt" needs pre-exported .trt engines)
    "pytorch",
]
metrics = [  # metrics to evaluate
    "iou",
    "latency",
]

basepath = os.path.dirname(__file__)
# Evaluate both the single-frame base models (egopath/weights) and the temporal
# RNN models (egopathrnn/weights). "weights" is kept for backward compatibility.
weight_roots = [
    os.path.join(basepath, "egopath", "weights"),
    os.path.join(basepath, "egopathrnn", "weights"),
    os.path.join(basepath, "weights"),
]
models_to_eval = []  # (display_name, model_path, is_rnn)
for root in weight_roots:
    if not os.path.isdir(root):
        continue
    for name in sorted(os.listdir(root)):
        model_path = os.path.join(root, name)
        if not os.path.isdir(model_path) or not os.path.exists(
            os.path.join(model_path, "config.yaml")
        ):
            continue
        is_rnn = "egopathrnn" in root or name.endswith("RNN")
        models_to_eval.append((name, model_path, is_rnn))

if "iou" in metrics:
    with open(os.path.join("configs", "global.yaml")) as f:
        config = yaml.safe_load(f)
    images_path = config["images_path"]
    annotations_path = config["annotations_path"]
    set_seeds(config["seed"])
    with open(annotations_path) as json_file:
        indices = list(range(len(json.load(json_file).keys())))
    random.shuffle(indices)
    proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    train_indices, val_indices, test_indices = split_dataset(indices, proportions)
    test_dataset = PathsDataset(
        images_path, annotations_path, test_indices, config, "segmentation"
    )

stats = []
for model, model_path, is_rnn in models_to_eval:
    with open(os.path.join(model_path, "config.yaml")) as f:
        model_config = yaml.safe_load(f)
    method = model_config["method"]
    backbone = model_config["backbone"]
    if backbone not in backbones or method not in methods:
        continue
    mtype = "rnn" if is_rnn else "single"
    for runtime in runtimes:
        try:
            latency = iou = None
            if "latency" in metrics:
                time.sleep(30)  # cooldown for stable timing
                latency = LatencyEvaluator(
                    model_path=model_path, runtime=runtime, device=device
                ).evaluate()
            if "iou" in metrics:
                iou = IoUEvaluator(
                    dataset=test_dataset, model_path=model_path, runtime=runtime, device=device
                ).evaluate()
        except Exception as e:  # noqa: BLE001 - skip a broken model/runtime, keep going
            print(f"[skip] {model} ({mtype}, {runtime}): {type(e).__name__}: {e}")
            continue
        precision = "amx" if runtime == "tensorrt" else "fp32"
        print(f"[done] {model} ({mtype}, {runtime})"
              + (f" latency={latency * 1000:.2f}ms" if latency is not None else "")
              + (f" iou={iou:.5f}" if iou is not None else ""))
        stats.append(
            f"{runtime},{backbone},{precision},{method},{mtype},{model}"
            + (f",{latency * 1000:.2f}" if "latency" in metrics else "")
            + (f",{iou:.5f}" if "iou" in metrics else "")
            + "\n"
        )
stats.sort()

os.makedirs("output", exist_ok=True)
with open(os.path.join("output", "eval.csv"), "w") as f:
    f.write("runtime,backbone,precision,method,type,model")
    f.write(",latency" if "latency" in metrics else "")
    f.write(",iou" if "iou" in metrics else "")
    f.write("\n")
    f.writelines(stats)
