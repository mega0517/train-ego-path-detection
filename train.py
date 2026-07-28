import argparse
import json
import os
import random
import re

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import wandb
import yaml

from src.nn.loss import (
    BinaryDiceLoss,
    CrossEntropyLoss,
    MultiHypothesisRegressionLoss,
    TrainEgoPathRegressionLoss,
)
from src.nn.model import (
    MultiPathRegressionNet,
    ClassificationNet,
    ClassificationNetRNN,
    RegressionNet,
    RegressionNetRNN,
    SegmentationNet,
    SegmentationNetRNN,
)
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset
from src.utils.dataset import (
    PathsDataset,
    RealSequencePathsDataset,
    SequencePathsDataset,
    gpu_collate_fn,
)
from src.utils.evaluate import IoUEvaluator
from src.utils.gpu_transforms import GpuPreprocess
from src.utils.trainer import train

# Speed-optimized run: trade bit-exact reproducibility for throughput. Seeds are
# still set, but kernels may be nondeterministic and cudnn auto-tunes algorithms.
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Training Script")
    parser.add_argument(
        "method",
        type=str,
        choices=["regression", "classification", "segmentation"],
        help="Method to use for the prediction head ('regression', 'classification' or 'segmentation').",
    )
    parser.add_argument(
        "backbone",
        type=str,
        choices=[f"resnet{x}" for x in [18, 34, 50]]
        + [f"efficientnet-b{x}" for x in [0, 1, 2, 3]],
        help="Backbone to use (e.g., 'resnet18', 'efficientnet-b3').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use ('cpu', 'cuda', 'cuda:x' or 'mps').",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Train the temporal RNN variant that tracks the ego-path over a sequence of frames.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Name of the trained per-frame model in weights/ to build the temporal model on (required with --temporal). The temporal model is saved as '<base-model>RNN'.",
    )
    parser.add_argument(
        "--finetune-base",
        action="store_true",
        help="With --temporal, also fine-tune the base per-frame weights instead of only training the RNN (base is frozen by default).",
    )
    parser.add_argument(
        "--n-hypotheses",
        type=int,
        default=1,
        help="Regression only: predict this many ego-path hypotheses plus a score per"
        " hypothesis, trained winner-takes-all. 1 (default) keeps the single-path model."
        " Use >1 where the path is genuinely ambiguous (facing switches), so the model"
        " can keep both branches instead of regressing between them.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Warm-start from a trained single-path model in egopath/weights (e.g."
        " chromatic-laughter-5); its final layer is replicated per hypothesis.",
    )
    parser.add_argument("--wta-epsilon", type=float, default=0.05,
                        help="Weight kept on the losing hypotheses (0 = pure WTA, which"
                             " tends to starve and kill the losing heads).")
    parser.add_argument("--score-weight", type=float, default=0.1,
                        help="Weight of the hypothesis-score cross-entropy term.")
    parser.add_argument(
        "--seq-stride",
        type=int,
        default=None,
        help="With --temporal, build sequences from REAL consecutive video frames spaced this"
        " many frames apart (e.g. 10 on a 10 fps recording = 1 fps), instead of the"
        " synthetic crop-interpolated pseudo-sequences. Requires a dataset recorded as"
        " video with <scene>__<index>_<timestamp> filenames (e.g. OSDaR23).",
    )
    parser.add_argument(
        "--seq-occlusion",
        type=float,
        default=None,
        help="With --temporal, probability of masking the bottom band of the last frame's input (target kept), forcing the RNN to recover the hidden path from past frames (sets seq_occlusion_prob).",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="With --temporal, suffix appended to the saved run name '<base-model>RNN<suffix>' (e.g. '-occ') so variants do not overwrite each other.",
    )
    parser.add_argument(
        "--gpu-preprocess",
        action="store_true",
        help="Offload JPEG decode/crop/resize/jitter/flip to the GPU (temporal training only). Dataloader workers only read raw file bytes, freeing CPU cores.",
    )
    parser.add_argument(
        "--prior-channel",
        action="store_true",
        help="Append a 4th input channel holding the previously accepted ego-path, so"
             " the branch the model committed to survives the frames where the switch"
             " geometry that decided it has left the field of view. The channel enters"
             " at the first conv with zero-initialised weights, so the run starts as"
             " the per-frame baseline. See PathsDataset.generate_prior for how the"
             " prior is corrupted during training.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the number of epochs from the method config.")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override the learning rate from the method config.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the batch size from the method config.")
    parser.add_argument("--images-path", type=str, default=None,
                        help="Override the images directory from the global config.")
    parser.add_argument("--annotations-path", type=str, default=None,
                        help="Override the annotations JSON from the global config.")
    parser.add_argument("--group-regex", type=str, default=None,
                        help="Keep frames of one recording in the same split. The first "
                             "capture group of this regex applied to the filename names "
                             r"the recording, e.g. '^(.*?)__' for OSDaR23.")
    parser.add_argument("--group-split", type=str, default="balanced",
                        choices=["balanced", "diverse"],
                        help="With --group-regex, how to hand recordings to the splits. "
                             "'balanced' packs the longest first and hits the quotas "
                             "closely; 'diverse' spends the eval quota on the shortest "
                             "recordings so val/test span many places.")
    parser.add_argument("--save-from", type=float, default=0.9,
                        help="Fraction of training after which best.pt may be saved "
                             "(default 0.9, the upstream behaviour). Use 0 to keep the "
                             "global validation optimum, which matters once the run "
                             "overfits before the final epochs.")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all visible CUDA GPUs via DataParallel (splits the batch).")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume an interrupted run from its full-state checkpoint. "
                             "Bare --resume uses <save_path>/last.pt (works for --temporal, "
                             "whose save path is deterministic); or pass an explicit "
                             "path/to/last.pt and training continues in that directory. "
                             "Launch with the same arguments as the interrupted run.")
    return parser.parse_args()


def load_base_weights(base_net, ckpt_path, device):
    """Loads per-frame weights into a base net, stripping any torch.compile prefix."""
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}
    base_net.load_state_dict(state)


def main(args):
    method = args.method
    device = torch.device(args.device)
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)

    with open(os.path.join(base_path, "configs", "global.yaml")) as f:
        global_config = yaml.safe_load(f)

    # A temporal model wraps a specific trained base, so its method and architecture
    # are dictated by that base; the CLI method/backbone are then ignored.
    base_config = None
    if args.temporal:
        if args.base_model is None:
            raise ValueError("--base-model is required with --temporal")
        base_cfg_path = os.path.join(
            base_path, "egopath", "weights", args.base_model, "config.yaml"
        )
        with open(base_cfg_path) as f:
            base_config = yaml.safe_load(f)
        method = base_config["method"]
        if args.method != method or args.backbone != base_config["backbone"]:
            logger.info(
                f"\n[temporal] Inheriting base architecture from {args.base_model} "
                f"(method={method}, backbone={base_config['backbone']}); "
                f"CLI method/backbone are ignored."
            )

    with open(os.path.join(base_path, "configs", f"{method}.yaml")) as f:
        method_config = yaml.safe_load(f)
    config = {
        **global_config,
        **method_config,
        "method": method,
        "backbone": args.backbone,
    }
    if args.temporal:
        # training/data/augmentation knobs come from the current config files;
        # the model architecture is taken from the base model so its weights load.
        for key in (
            "backbone", "input_shape", "anchors", "pool_channels",
            "fc_hidden_size", "classes", "decoder_channels",
        ):
            if key in base_config:
                config[key] = base_config[key]
        config["temporal"] = True
        config["base_model"] = args.base_model
        config["freeze_base"] = not args.finetune_base

    if args.prior_channel:
        if args.temporal:
            raise ValueError(
                "--prior-channel and --temporal are alternative ways to carry state;"
                " the prior channel replaces the RNN refiner rather than stacking on it"
            )
        config["input_shape"] = [4, *config["input_shape"][1:]]
        config["prior_channel"] = True

    if args.n_hypotheses > 1:
        if method != "regression" or args.temporal:
            raise ValueError("--n-hypotheses is implemented for non-temporal regression")
        config["n_hypotheses"] = args.n_hypotheses
        config["wta_epsilon"] = args.wta_epsilon
        config["score_weight"] = args.score_weight
    if args.temporal and args.seq_stride is not None:
        config["seq_stride"] = args.seq_stride
        config["real_sequences"] = True
    if args.temporal and args.seq_occlusion is not None:
        config["seq_occlusion_prob"] = args.seq_occlusion

    # GPU-side preprocessing is only wired for the temporal (sequence) dataset.
    config["gpu_preprocess"] = (
        bool(args.gpu_preprocess) and args.temporal and args.seq_stride is None
    )
    if args.gpu_preprocess and args.seq_stride is not None:
        print("\n[gpu-preprocess] disabled: real-sequence training decodes several"
              " frames per sample on the CPU dataloader.")
    if args.gpu_preprocess and not args.temporal:
        logger.info(
            "\n[gpu-preprocess] --gpu-preprocess only applies to --temporal training; "
            "ignoring for the per-frame run."
        )

    # Optional per-run hyperparameter overrides (used by the web training UI).
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.images_path is not None:
        config["images_path"] = args.images_path
    if args.annotations_path is not None:
        config["annotations_path"] = args.annotations_path

    set_seeds(config["seed"])  # set random state
    with open(config["annotations_path"]) as json_file:
        names = sorted(json.load(json_file).keys())
    proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    if args.group_regex:
        # Frames from one recording are near-duplicates, so splitting them
        # individually leaks the validation set into training and inflates the
        # score. Keep every frame of a recording on the same side of the split.
        pattern = re.compile(args.group_regex)
        groups = {}
        for i, name in enumerate(names):
            m = pattern.search(name)
            groups.setdefault(m.group(1) if m and m.groups() else name, []).append(i)
        keys = sorted(groups)
        random.shuffle(keys)
        # Recordings differ in length by an order of magnitude, so balance the
        # splits by FRAME count: place the largest recordings first, each into
        # whichever split is furthest below its quota. Filling the splits in turn
        # instead would starve whichever one is filled last.
        targets = [p * len(names) for p in proportions]
        parts, filled = [[], [], []], [0.0, 0.0, 0.0]
        if args.group_split == "diverse":
            # Spend the eval quota on the SHORTEST recordings, so validation and
            # test cover many places instead of one long one. A test set that is
            # 210 near-identical frames of a single platform reports how well the
            # model does there, not whether it generalises.
            keys.sort(key=lambda k: (len(groups[k]), k))
            for key in keys:
                short = [p for p in (1, 2) if filled[p] < targets[p]]
                part = min(short, key=lambda p: filled[p] / targets[p]) if short else 0
                parts[part].append(key)
                filled[part] += len(groups[key])
        else:
            keys.sort(key=lambda k: -len(groups[k]))
            for key in keys:
                part = min((p for p in range(3) if targets[p] > 0),
                           key=lambda p: filled[p] / targets[p])
                parts[part].append(key)
                filled[part] += len(groups[key])
        train_indices, val_indices, test_indices = (
            sorted(i for k in p for i in groups[k]) for p in parts
        )
        print(f"Grouped split on /{args.group_regex}/ ({args.group_split}): "
              f"{len(keys)} groups -> train {len(train_indices)}, "
              f"val {len(val_indices)}, test {len(test_indices)} frames")
        print(f"  val groups : {sorted(parts[1])}")
        print(f"  test groups: {sorted(parts[2])}")
    else:
        indices = list(range(len(names)))
        random.shuffle(indices)
        train_indices, val_indices, test_indices = split_dataset(indices, proportions)
    set_seeds(config["seed"])  # reset random state

    real_seq = args.temporal and args.seq_stride is not None
    dataset_cls = (
        RealSequencePathsDataset if real_seq
        else SequencePathsDataset if args.temporal
        else PathsDataset
    )
    seq_kwargs = (
        {"seq_len": config["seq_len"], "seq_jitter": config["seq_jitter"]}
        if args.temporal
        else {"prior": True} if args.prior_channel
        else {}
    )
    if real_seq:
        seq_kwargs["seq_stride"] = config["seq_stride"]
    train_dataset = dataset_cls(
        imgs_path=config["images_path"],
        annotations_path=config["annotations_path"],
        indices=train_indices,
        config=config,
        method=method,
        img_aug=True,
        to_tensor=True,
        **seq_kwargs,
    )
    val_dataset = (
        dataset_cls(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=val_indices,
            config=config,
            method=method,
            img_aug=True,
            to_tensor=True,
            **seq_kwargs,
        )
        if len(val_indices) > 0
        else None
    )
    # With gpu_preprocess on, workers yield variable-length raw JPEG byte tensors
    # which the default collate cannot stack; pin_memory is also unhelpful for the
    # ragged byte list, so it is disabled on that path.
    gpu_preprocess = config.get("gpu_preprocess", False)
    if gpu_preprocess:
        # Passing many raw-byte tensors across workers exhausts the default
        # file_descriptor IPC strategy ("Bad file descriptor"); file_system shares
        # via temp files and handles the ragged byte payloads reliably.
        torch.multiprocessing.set_sharing_strategy("file_system")
    collate_fn = gpu_collate_fn if gpu_preprocess else None
    pin_memory = not gpu_preprocess
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=pin_memory,
        persistent_workers=config["workers"] > 0,
        prefetch_factor=4 if config["workers"] > 0 else None,
        worker_init_fn=set_worker_seeds,
        generator=torch.Generator().manual_seed(config["seed"]),
        collate_fn=collate_fn,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            pin_memory=pin_memory,
            persistent_workers=config["workers"] > 0,
            prefetch_factor=4 if config["workers"] > 0 else None,
            worker_init_fn=set_worker_seeds,
            generator=torch.Generator().manual_seed(config["seed"]),
            collate_fn=collate_fn,
        )
        if val_dataset is not None
        else None
    )

    if method == "regression":
        if args.temporal:
            model = RegressionNetRNN(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
                seq_len=config["seq_len"],
                rnn_hidden=config["rnn_hidden"],
                rnn_layers=config["rnn_layers"],
            ).to(device)
        elif args.n_hypotheses > 1:
            model = MultiPathRegressionNet(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
                n_hypotheses=config["n_hypotheses"],
                pretrained=config["pretrained"],
            )
            if args.init_from:
                ckpt = os.path.join(
                    base_path, "egopath", "weights", args.init_from, "best.pt"
                )
                copied = model.load_single_path_state(
                    torch.load(ckpt, map_location="cpu")
                )
                logger.info(f"\n[multi-path] warm-started {copied} tensors from {args.init_from}")
            model = model.to(device)
        else:
            model = RegressionNet(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
                pretrained=config["pretrained"],
            ).to(device)
    elif method == "classification":
        if args.temporal:
            model = ClassificationNetRNN(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                classes=config["classes"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
                seq_len=config["seq_len"],
                rnn_hidden=config["rnn_hidden"],
                rnn_layers=config["rnn_layers"],
            ).to(device)
        else:
            model = ClassificationNet(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                classes=config["classes"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
                pretrained=config["pretrained"],
            ).to(device)
    elif method == "segmentation":
        if args.temporal:
            model = SegmentationNetRNN(
                backbone=config["backbone"],
                decoder_channels=tuple(config["decoder_channels"]),
                seq_len=config["seq_len"],
                rnn_hidden=config["seg_rnn_hidden"],
                rnn_layers=config["rnn_layers"],
            ).to(device)
        else:
            model = SegmentationNet(
                backbone=config["backbone"],
                decoder_channels=tuple(config["decoder_channels"]),
                pretrained=config["pretrained"],
                in_channels=config["input_shape"][0],
            ).to(device)
    else:
        raise ValueError

    if args.temporal:
        base_ckpt = os.path.join(
            base_path, "egopath", "weights", args.base_model, "best.pt"
        )
        load_base_weights(model.base, base_ckpt, device)
        logger.info(f"\nLoaded base per-frame weights from {base_ckpt}")
        if config["freeze_base"]:
            for param in model.base.parameters():
                param.requires_grad = False
            model.base.eval()

    # Multi-GPU via DataParallel (batch split across all visible GPUs). Done after
    # base load/freeze so those still touch the raw model. torch.compile is skipped
    # in this path because compiling a DataParallel wrapper is unreliable.
    use_multi = args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_multi:
        logger.info(f"\n[multi-gpu] DataParallel over {torch.cuda.device_count()} GPUs "
                    f"(batch {config['batch_size']} split across them).")
        model = torch.nn.DataParallel(model)
    else:
        if args.multi_gpu:
            logger.info("\n[multi-gpu] requested but <2 GPUs visible; using a single GPU.")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}. Running model without compilation.")

    # Run W&B offline by default so no account/login is required. Set
    # WANDB_MODE=online (and log in) to sync to the cloud dashboard instead.
    os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(
        project="train-ego-path-detection",
        config=config,
        dir=os.path.join(base_path),
        mode=os.environ["WANDB_MODE"],
    )
    if args.temporal:
        save_path = os.path.join(
            base_path, "egopathrnn", "weights", f"{args.base_model}RNN{args.run_suffix}"
        )
    else:
        run_name = wandb.run.name or wandb.run.id or f"{method}-{args.backbone}"
        save_path = os.path.join(base_path, "weights", run_name)

    # Resolve --resume. An explicit checkpoint path pins save_path to its directory
    # so the resumed run keeps writing to the same place (a non-temporal run would
    # otherwise get a fresh random wandb name). Bare --resume ("auto") targets the
    # deterministic temporal save path.
    resume_from = None
    if args.resume is not None:
        if args.resume == "auto":
            resume_from = os.path.join(save_path, "last.pt")
        else:
            resume_from = os.path.abspath(args.resume)
            save_path = os.path.dirname(resume_from)

    logger.info(f"\nSaving model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    if method == "regression":
        criterion = TrainEgoPathRegressionLoss(
            ylimit_loss_weight=config["ylimit_loss_weight"],
            perspective_weight_limit=train_dataset.get_perspective_weight_limit(
                percentile=config["perspective_weight_limit_percentile"],
                logger=logger,
            )
            if config["perspective_weight_limit_percentile"] is not None
            else None,
        )
        if config["perspective_weight_limit_percentile"] is not None:
            set_seeds(config["seed"])  # reset random state
        if args.n_hypotheses > 1:
            criterion = MultiHypothesisRegressionLoss(
                criterion,
                n_hypotheses=config["n_hypotheses"],
                epsilon=config["wta_epsilon"],
                score_weight=config["score_weight"],
            )
    elif method == "classification":
        criterion = CrossEntropyLoss()
    elif method == "segmentation":
        criterion = BinaryDiceLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )
    scheduler = None
    if config["scheduler"] == "one_cycle":
        total = config["epochs"]
        if total < 4:
            # OneCycleLR divides by each phase's length; a 1-cycle over 1-3 epochs
            # leaves the warmup or anneal phase 0 steps long (ZeroDivisionError) and
            # is meaningless anyway. Fall back to a constant LR.
            logger.info(f"\n[scheduler] one_cycle needs >=4 epochs (got {total}); "
                        f"using a constant learning rate instead.")
        else:
            # Keep the 0.1 warmup for normal runs, but for small epoch counts raise
            # pct_start just enough that both phases keep >=1 step.
            pct_start = min(max(0.1, 2.0 / total), 1.0 - 2.0 / total)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=config["learning_rate"],
                total_steps=total,
                pct_start=pct_start,
            )

    preprocess = GpuPreprocess(config, device) if gpu_preprocess else None
    if gpu_preprocess:
        logger.info("\n[gpu-preprocess] GPU-side decode/crop/resize/jitter/flip enabled.")

    logger.info(f"\nTraining {method} model for {config['epochs']} epochs...")
    train(
        epochs=config["epochs"],
        dataloaders=(train_loader, val_loader),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=save_path,
        device=device,
        logger=logger,
        val_iterations=config["val_iterations"],
        preprocess=preprocess,
        save_from=args.save_from,
        resume_from=resume_from,
    )

    if len(test_indices) > 0:
        logger.info("\nEvaluating on test set...")
        test_dataset = PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=test_indices,
            config=config,
            method="segmentation",
        )
        iou_evaluator = IoUEvaluator(
            dataset=test_dataset,
            model_path=save_path,
            runtime="pytorch",
            device=device,
        )
        test_iou = iou_evaluator.evaluate()
        logger.info(f"Test IoU: {test_iou:.5f}")
        wandb.log({"test_iou": test_iou})


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
