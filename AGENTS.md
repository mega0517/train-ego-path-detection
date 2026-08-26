# AI Agent Instructions

## Purpose
This repository implements TEP-Net for train ego-path detection using PyTorch, with support for classification, regression, and segmentation heads plus optional TensorRT inference.

## Key entry points
- `train.py` — train a model using a method and backbone defined by `configs/{method}.yaml` and `configs/global.yaml`
- `detect.py` — run inference on an image or video using a trained model directory under `weights/`
- `demo.py` — example inference script for local images and optional TensorRT usage
- `eval.py` — evaluate trained models on the test set
- `gui_app.py` — interactive GUI application for inference with file, model, device, and crop selection
- `launch_gui.sh` / `launch_gui.bat` — cross-platform launchers for the GUI application

## Configuration and model layout
- `configs/global.yaml` contains dataset paths, input shape, training hyperparameters, and common settings
- `configs/{regression,classification,segmentation}.yaml` contains method-specific model and training parameters
- Trained model directories in `weights/` contain `config.yaml` and `best.pt` (and optionally `best.trt` for TensorRT)

## Important conventions
- `src/nn/` holds model definitions and loss functions
- `src/utils/` holds dataset loading, training, evaluation, inference interface, and visualization helpers
- `src/utils/interface.py` defines `Detector`, which is the main inference wrapper for both PyTorch and TensorRT
- `Detector` expects `model_path`, `crop_coords`, `runtime`, and `device`
- Cropping modes: fixed tuple, `"auto"`, or `None`

## Common tasks and commands
- Install dependencies:
  - `pip install -r requirements.txt`
- Run training:
  - `python train.py regression resnet18 --device cuda`
- Run inference:
  - `python detect.py <model-name> <input> --output output --crop auto --device cuda`
- Run demo:
  - `python demo.py`
- Run evaluation:
  - `python eval.py`

## Known environment considerations
- TensorRT inference is optional and requires `tensorrt`, `pycuda`, and model conversion to `best.trt`
- Scripts set `PYTORCH_ENABLE_MPS_FALLBACK = 1` and may target `cuda`, `mps`, or `cpu`
- `train.py` calls `wandb.login()` before training; offline or CI use may need a valid W&B setup or environment override
- `train.py` attempts `torch.compile(model)` with fallback to standard execution if compilation fails

## Guidance for agents
- Preserve dataset paths and weight paths when editing training/inference code
- Avoid changing the core `Detector` interface unless there is a clear need for new runtime options
- When adding features, keep method-specific logic separate in `src/nn/` and `configs/`
- Use `README.md` as the source of truth for installation, dataset download, and model weight instructions

## References
- `README.md` — installation, demo, training, inference, evaluation, and dataset setup
- `configs/` — configuration files for reproducible training and inference
- `src/utils/interface.py` — inference interface and runtime handling
