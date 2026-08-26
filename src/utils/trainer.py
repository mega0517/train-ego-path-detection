import os
import time

import torch

import wandb


def train_epoch(
    model, criterion, device, dataloader, optimizer, use_amp=False, preprocess=None
):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for batch in dataloader:
        if preprocess is not None:
            # GPU-preprocess path: workers yield raw bytes + geometry; decode,
            # crop, resize, jitter and flip happen here on the GPU.
            data, target = preprocess(batch)
        else:
            data, *target = batch
            data = data.to(device)
            target = (
                [t.to(device) for t in target]
                if len(target) > 1
                else target[0].to(device)
            )
        model.zero_grad(set_to_none=True)
        # bf16 autocast on CUDA: same exponent range as fp32, so no GradScaler needed.
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches


def val_epoch(model, criterion, device, dataloader, use_amp=False, preprocess=None):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    # no_grad (not inference_mode): inference_mode marks tensors as inference
    # tensors, which breaks cuDNN RNN flatten_parameters() under DataParallel
    # ("Inplace update to inference tensor ...") during validation.
    with torch.no_grad():
        for batch in dataloader:
            if preprocess is not None:
                data, target = preprocess(batch)
            else:
                data, *target = batch
                data = data.to(device)
                target = (
                    [t.to(device) for t in target]
                    if len(target) > 1
                    else target[0].to(device)
                )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                output = model(data)
                loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / num_batches


def train(
    epochs,
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    save_path,
    device,
    logger,
    val_iterations=1,
    preprocess=None,
    save_from=0.9,
    resume_from=None,
):
    """Trains the model and saves the best weights.

    Args:
        epochs (int): Number of epochs to train.
        dataloaders (tuple): Tuple containing the training and validation dataloaders.
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function to use.
        optimizer (torch.nn.Module): Optimizer to use.
        scheduler (torch.nn.Module): Learning rate scheduler to use.
        save_path (str): Path to save the best model weights.
        device (torch.device): Device to use.
        logger (logging.Logger): Logger to use.
        val_iterations (int, optional): Number of validation epochs to average. Defaults to 1.
        save_from (float, optional): Fraction of training after which the best
            checkpoint may be saved. The default 0.9 ignores the first 90% of
            epochs, which is fine while validation loss keeps falling but discards
            the real optimum once the run starts overfitting. Pass 0.0 to keep the
            global best. Defaults to 0.9.
        resume_from (str, optional): Path to a ``last.pt`` checkpoint to resume
            from. Restores model/optimizer/scheduler/epoch/best-loss and continues
            at the next epoch. The run must be launched with the same arguments
            (architecture, epochs, multi-gpu) as the interrupted one. Defaults to None.
    """
    train_loader, val_loader = dataloaders
    # bf16 autocast only on CUDA; harmless no-op on cpu/mps.
    use_amp = device.type == "cuda"
    best_val_loss = float("inf")
    epoch = 0
    start_epoch = 0
    last_path = os.path.join(save_path, "last.pt")

    def save_checkpoint(completed_epoch):
        """Atomically write full training state so a crash mid-write can't corrupt it."""
        ckpt = {
            "epoch": completed_epoch,   # last fully completed epoch (0-indexed)
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
        }
        tmp = last_path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, last_path)

    if resume_from is not None and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        logger.info(
            f"\nResumed from {resume_from}: continuing at epoch {start_epoch + 1}/{epochs} "
            f"(best val loss so far {best_val_loss:.5f})."
        )
        if start_epoch >= epochs:
            logger.info("Checkpoint already reached the target epoch; nothing to do.")
            return
    elif resume_from is not None:
        logger.info(f"\n--resume given but no checkpoint at {resume_from}; starting fresh.")

    try:
        for epoch in range(start_epoch, epochs):
            train_loss = train_epoch(
                model,
                criterion,
                device,
                train_loader,
                optimizer,
                use_amp=use_amp,
                preprocess=preprocess,
            )
            val_loss = 0
            if val_loader is not None:
                # each validation epoch is unique due to data augmentation, so we can average multiple
                for _ in range(val_iterations):
                    val_loss += val_epoch(
                        model,
                        criterion,
                        device,
                        val_loader,
                        use_amp=use_amp,
                        preprocess=preprocess,
                    )
                val_loss /= val_iterations
            if scheduler is not None:
                scheduler.step()
            saved = epoch >= epochs * save_from and val_loss < best_val_loss
            if saved:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_path, "best.pt"))
            logger.info(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
                + f" | EPOCH {(epoch+1):0{len(str(epochs))}}/{epochs}"
                + f" | TRAIN LOSS: {train_loss:.5f}"
                + f" | VAL LOSS: {val_loss:.5f}"
                # mark the epoch best.pt came from; the log timestamps cannot be
                # matched against the file mtime on an NFS-mounted weights dir
                + (" | saved best.pt" if saved else "")
            )
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            # full-state checkpoint after every epoch so an interrupted run can
            # resume near where it stopped (see --resume).
            save_checkpoint(epoch)
    except KeyboardInterrupt:
        interrupted_path = os.path.join(save_path, "interrupted.pt")
        torch.save(model.state_dict(), interrupted_path)
        # Do NOT overwrite last.pt here: it already holds the last *completed*
        # epoch, whereas this interrupt likely landed mid-epoch. Resuming from
        # last.pt re-runs the interrupted epoch rather than skipping it.
        resume_hint = (f"resume with --resume {last_path}"
                       if os.path.isfile(last_path) else "no epoch finished yet; rerun from scratch")
        logger.info(
            f"\nTraining interrupted at epoch {epoch + 1}/{epochs}. "
            f"Saved current weights to {interrupted_path}; {resume_hint}"
        )
        raise
    wandb.log({"best_val_loss": best_val_loss})
