"""Training loop: AMP, gradient accumulation, early stopping, checkpoint/resume.

The loop is device-agnostic.  On CUDA it enables mixed precision
(``torch.amp.autocast`` + ``GradScaler``); on CPU it runs in float32.  Model
selection is by validation QWK by default.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluate import predict, summarise
from .utils import ensure_dir, get_logger, progress, save_json, supports_amp

logger = get_logger()


def optimizer_update_succeeded(old_scale: float, new_scale: float) -> bool:
    """GradScaler lowers its scale exactly when it skipped an optimizer step."""
    return new_scale >= old_scale


# --------------------------------------------------------------------------- #
# Optimizer / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    no_decay_tokens = ("bias", "LayerNorm", "layer_norm", "norm")
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in no_decay_tokens):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if not decay:
        groups = groups[1:]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, kind: str = "linear"):
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    warmup_steps = max(0, min(warmup_steps, max(total_steps - 1, 0)))
    if kind == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if kind == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if kind in {"none", "constant"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"unknown scheduler: {kind!r}")


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        *,
        device,
        loss_computer,
        decode_mode: str = "argmax",
        out_dir: str | Path = "results/run",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        epochs: int = 3,
        grad_accumulation: int = 1,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        selection_metric: str = "qwk",
        early_stopping_patience: int = 2,
        scheduler: str = "linear",
        log_every: int = 50,
        max_train_steps: Optional[int] = None,
        save_total_limit: int = 2,
        save_epoch_checkpoints: bool = True,
    ):
        self.model = model
        self.device = device
        self.loss_computer = loss_computer
        self.decode_mode = decode_mode
        self.out_dir = ensure_dir(out_dir)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.epochs = epochs
        self.grad_accumulation = max(1, int(grad_accumulation))
        self.max_grad_norm = max_grad_norm
        self.fp16 = bool(fp16) and supports_amp(device)
        self.selection_metric = selection_metric
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_kind = scheduler
        self.log_every = log_every
        self.max_train_steps = max_train_steps
        self.save_total_limit = max(1, int(save_total_limit))
        self.save_epoch_checkpoints = bool(save_epoch_checkpoints)

        self.optimizer = None
        self.scheduler_obj = None
        self.scaler = None
        self.history: list[dict] = []
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = -math.inf
        self.best_epoch = -1
        self.best_model_state_dict = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _setup_optimizer(self, train_loader: DataLoader) -> None:
        self.optimizer = build_optimizer(self.model, self.learning_rate, self.weight_decay)
        steps_per_epoch = math.ceil(len(train_loader) / self.grad_accumulation)
        total_steps = steps_per_epoch * self.epochs
        if self.max_train_steps:
            total_steps = min(total_steps, self.max_train_steps)
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.total_steps = total_steps
        self.scheduler_obj = build_scheduler(
            self.optimizer, warmup_steps, total_steps, self.scheduler_kind
        )
        if self.fp16:
            try:
                self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=True)
            except TypeError:  # older torch
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        *,
        include_optimizer: bool = True,
    ) -> bool:
        """Write a checkpoint atomically.

        Returns ``False`` when the filesystem rejects the write (for example
        when a Colab/Kaggle volume is full) instead of losing the epoch: the
        training run can still finish and save its predictions/metrics.
        """
        path = Path(path)
        ensure_dir(path.parent)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if (self.optimizer and include_optimizer) else None
            ),
            "scheduler_state_dict": (
                self.scheduler_obj.state_dict()
                if (self.scheduler_obj and include_optimizer)
                else None
            ),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "selection_metric": self.selection_metric,
        }
        tmp = path.with_name(path.name + ".tmp")
        try:
            torch.save(state, tmp)
            os.replace(tmp, path)
            return True
        except (OSError, RuntimeError) as exc:
            tmp.unlink(missing_ok=True)
            logger.warning(
                "Could not save checkpoint %s (%s). Free disk space under %s "
                "or set train.cleanup_checkpoints_after_run=true; continuing.",
                path,
                exc,
                path.parent,
            )
            return False

    def load_checkpoint(self, path: str | Path, train_loader: Optional[DataLoader] = None) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.start_epoch = int(state.get("epoch", -1)) + 1
        self.global_step = int(state.get("global_step", 0))
        self.best_metric = float(state.get("best_metric", -math.inf))
        self.best_epoch = int(state.get("best_epoch", -1))
        self.history = state.get("history", [])
        if train_loader is not None:
            self._setup_optimizer(train_loader)
            if state.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if state.get("scheduler_state_dict") and self.scheduler_obj:
                self.scheduler_obj.load_state_dict(state["scheduler_state_dict"])
        logger.info(
            "Resumed from %s at epoch %d (best_metric=%.4f)",
            path,
            self.start_epoch,
            self.best_metric,
        )

    def _prune_checkpoints(self, keep: int | None = None) -> None:
        limit = self.save_total_limit if keep is None else keep
        checkpoints = sorted(
            self.out_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(checkpoints) > limit:
            old = checkpoints.pop(0)
            if old.name != "checkpoint_best.pt":
                old.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Train / validate
    # ------------------------------------------------------------------ #
    def _metric_value(self, metrics: dict) -> float:
        if self.selection_metric == "qwk":
            return float(metrics.get("qwk", float("nan")))
        if self.selection_metric == "qwk_minus_mae":
            return float(metrics.get("qwk", 0.0) - metrics.get("mae", 0.0))
        if self.selection_metric == "loss":
            return -float(metrics.get("loss", float("inf")))
        raise ValueError(f"unknown selection metric: {self.selection_metric!r}")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        running_loss = 0.0
        seen = 0
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        iterator = progress(
            train_loader,
            desc=f"train epoch {epoch + 1}/{self.epochs}",
            unit="batch",
            position=1,
        )
        for step, batch in enumerate(iterator):
            batch = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            accumulation_start = (step // self.grad_accumulation) * self.grad_accumulation
            accumulation_size = min(
                self.grad_accumulation, len(train_loader) - accumulation_start
            )
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.fp16
            ):
                outputs = self.model(batch)
                loss, _ = self.loss_computer(outputs["logits"], batch["labels"])
                scaled_loss = loss / accumulation_size

            if self.fp16:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % self.grad_accumulation == 0 or (step + 1) == len(train_loader):
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.fp16:
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # GradScaler skips optimizer.step() on non-finite gradients
                    # and reduces its scale. The LR schedule must advance only
                    # when weights were actually updated.
                    optimizer_updated = optimizer_update_succeeded(
                        old_scale, self.scaler.get_scale()
                    )
                else:
                    self.optimizer.step()
                    optimizer_updated = True
                if self.scheduler_obj is not None and optimizer_updated:
                    self.scheduler_obj.step()
                self.optimizer.zero_grad(set_to_none=True)
                if optimizer_updated:
                    self.global_step += 1

                if self.max_train_steps and self.global_step >= self.max_train_steps:
                    break

            running_loss += float(loss.detach()) * len(batch["labels"])
            seen += len(batch["labels"])

            # Refresh the postfix only at report cadence; calling it every
            # batch forces a tqdm redraw, which floods non-interactive logs.
            if self.log_every and (step + 1) % self.log_every == 0:
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        loss=f"{running_loss / max(seen, 1):.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.1e}",
                    )
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch %d step %d/%d loss=%.4f lr=%.2e",
                    epoch,
                    step + 1,
                    len(train_loader),
                    running_loss / max(seen, 1),
                    lr,
                )

        if hasattr(iterator, "close"):
            iterator.close()

        elapsed = time.perf_counter() - started
        result = {
            "epoch": epoch,
            "split": "train",
            "loss": running_loss / max(seen, 1),
            "seconds": elapsed,
            "n": seen,
        }
        if torch.cuda.is_available():
            result["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        logger.info("[train] epoch %d loss=%.4f time=%.1fs", epoch, result["loss"], elapsed)
        return result

    def validate(self, val_loader: DataLoader, epoch: int) -> dict:
        preds = predict(
            self.model,
            val_loader,
            self.device,
            decode_mode=self.decode_mode,
            use_amp=self.fp16,
            desc=f"val epoch {epoch + 1}/{self.epochs}",
            progress_position=1,
        )
        from .labels import NUM_CLASSES
        from .metrics import compute_metrics

        metrics = compute_metrics(preds["true_score"], preds["predicted_score"])
        result = {
            "epoch": epoch,
            "split": "val",
            "loss": float("nan"),
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        }
        logger.info("[val] epoch %d %s", epoch, summarise({"overall": metrics}))
        return result

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume_from: Optional[str] = None) -> dict:
        # AMP requires FP32 master weights. Some pretrained checkpoints are
        # distributed in FP16, so normalise before moving to the GPU.
        self.model.float()
        self.model.to(self.device)
        if resume_from:
            self.load_checkpoint(resume_from, train_loader)
        if self.optimizer is None:
            self._setup_optimizer(train_loader)

        epochs_without_improvement = 0
        history_path = self.out_dir / "history.json"
        csv_path = self.out_dir / "history.csv"

        epoch_bar = progress(
            range(self.start_epoch, self.epochs),
            desc="epochs",
            unit="epoch",
            position=0,
            leave=True,
        )
        for epoch in epoch_bar:
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            record = {
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "epoch"},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "epoch"},
                "epoch": epoch,
            }
            self.history.append(record)
            save_json(self.history, history_path)
            _append_csv(csv_path, record)

            score = self._metric_value(val_metrics)
            improved = np.isfinite(score) and score > self.best_metric
            if improved:
                self.best_metric = score
                self.best_epoch = epoch
                # Keep a CPU copy so final evaluation remains faithful even if
                # the output volume rejects the checkpoint write.
                self.best_model_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
                # Best checkpoints are only used for final evaluation, so omit
                # the optimizer state (saves ~1.2 GB per run).
                self.save_checkpoint(
                    self.out_dir / "checkpoint_best.pt",
                    epoch,
                    include_optimizer=False,
                )
                logger.info("New best %s=%.4f at epoch %d", self.selection_metric, score, epoch)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No improvement (%s=%.4f, best=%.4f) -- %d/%d epochs without improvement",
                    self.selection_metric,
                    score,
                    self.best_metric,
                    epochs_without_improvement,
                    self.early_stopping_patience,
                )

            # Make room *before* writing the next full checkpoint, otherwise a
            # full volume fails the save halfway through.
            if self.save_epoch_checkpoints:
                self._prune_checkpoints(keep=max(self.save_total_limit - 1, 0))
                self.save_checkpoint(self.out_dir / f"checkpoint_epoch_{epoch}.pt", epoch)
                self._prune_checkpoints()

            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(
                    best=f"{self.best_metric:.4f}",
                    best_epoch=self.best_epoch,
                )

            if (
                self.early_stopping_patience is not None
                and self.early_stopping_patience >= 0
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                logger.info("Early stopping triggered at epoch %d", epoch)
                if hasattr(epoch_bar, "update"):
                    epoch_bar.update(1)
                break

            if self.max_train_steps and self.global_step >= self.max_train_steps:
                logger.info("Reached max_train_steps=%d; stopping", self.max_train_steps)
                if hasattr(epoch_bar, "update"):
                    epoch_bar.update(1)
                break

        if hasattr(epoch_bar, "close"):
            epoch_bar.close()

        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)

        summary = {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "selection_metric": self.selection_metric,
            "total_steps": self.global_step,
            "history": self.history,
        }
        save_json(summary, self.out_dir / "train_summary.json")
        return summary


def _append_csv(path: Path, record: dict) -> None:
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
