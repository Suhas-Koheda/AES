"""Class weighting and cross-entropy for the retained holistic score head."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import NUM_CLASSES


def compute_class_weights(label_counts, num_classes: int = NUM_CLASSES, mode: str = "none"):
    """Return sklearn-style balanced class weights, or ``None``."""
    if mode in {None, "none", "None"}:
        return None
    if mode != "balanced":
        raise ValueError(f"unknown class weighting mode: {mode!r}")
    counts = torch.as_tensor(label_counts, dtype=torch.float64)
    if counts.numel() != num_classes:
        raise ValueError(f"expected {num_classes} counts, got {counts.numel()}")
    return (counts.sum() / (num_classes * counts.clamp(min=1.0))).float()


class LossComputer:
    """Six-class cross-entropy loss with optional label smoothing."""

    def __init__(self, class_weights=None, label_smoothing: float = 0.0):
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor):
        if logits.ndim != 2 or logits.size(-1) != NUM_CLASSES:
            raise ValueError(f"expected logits with shape (N, {NUM_CLASSES})")
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        return loss, {"ce": float(loss.detach())}
