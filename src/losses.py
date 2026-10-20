"""Loss functions.

Two objectives are supported and kept mathematically consistent with their
decoding rules:

1. **Softmax cross-entropy** on 6 logits.  Decoding is argmax (or expected
   score, if configured).

2. **CORAL ordinal loss** (Cao et al., 2020) on ``K-1 = 5`` ordered logits.
   The target is the extended binary vector ``y_k = 1[score > k]`` and the loss
   is the mean binary cross-entropy over the five cutpoints.  The predicted
   class is the number of cutpoints with ``sigmoid(logit_k) > 0.5``, i.e. the
   number of positive logits.  Crucially, CORAL logits are **not** passed
   through a softmax.

3. **Combined CE + expected-score regression** on the 6-way softmax head.
   ``E[score] = sum_k p_k (k+1)``; a smooth-L1 penalty pulls it toward the
   human score.  This is applied to genuine softmax probabilities, never to
   CORAL logits.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .labels import NUM_CLASSES


# --------------------------------------------------------------------------- #
# Class weighting
# --------------------------------------------------------------------------- #
def compute_class_weights(label_counts, num_classes: int = NUM_CLASSES, mode: str = "none"):
    """Return a ``(num_classes,)`` float tensor or ``None``.

    ``mode="balanced"`` uses ``N / (K * n_k)`` (sklearn's "balanced").
    """
    if mode in {None, "none", "None"}:
        return None
    if mode != "balanced":
        raise ValueError(f"unknown class weighting mode: {mode!r}")
    counts = torch.as_tensor(label_counts, dtype=torch.float64)
    if counts.numel() != num_classes:
        raise ValueError(f"expected {num_classes} counts, got {counts.numel()}")
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(torch.float32)


# --------------------------------------------------------------------------- #
# CORAL
# --------------------------------------------------------------------------- #
def coral_targets(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Extended binary targets ``y_k = 1[label > k]`` for k = 0..K-2."""
    if labels.dim() != 1:
        raise ValueError("labels must be 1-D")
    cutpoints = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > cutpoints.unsqueeze(0)).to(torch.float32)


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CORAL binary-cross-entropy loss.

    ``logits``: ``(N, K-1)`` ordered logits (no softmax).
    ``labels``: ``(N,)`` integer class indices in ``[0, K-1]``.
    """
    if logits.dim() != 2 or logits.size(-1) != NUM_CLASSES - 1:
        raise ValueError(
            f"CORAL logits must be (N, {NUM_CLASSES - 1}), got {tuple(logits.shape)}"
        )
    targets = coral_targets(labels, num_classes=NUM_CLASSES)
    if class_weights is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    # Weight each cutpoint by the weight of the "positive" class it separates.
    # Weight of cutpoint k is the mean of class weights for y>k vs y<=k.
    weights = class_weights.to(logits.device)
    cut_weights = torch.stack(
        [weights[k + 1 :].sum() / max(NUM_CLASSES - 1 - k, 1) for k in range(NUM_CLASSES - 1)]
    )
    cut_weights = cut_weights / cut_weights.mean().clamp(min=1e-9)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_cut = loss.mean(dim=0)
    return (per_cut * cut_weights).mean()


# --------------------------------------------------------------------------- #
# Softmax + expected-score regression
# --------------------------------------------------------------------------- #
def expected_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """``E[score] = sum_k softmax(logits)_k * (k + 1)`` on the 1..6 scale."""
    if logits.size(-1) != NUM_CLASSES:
        raise ValueError(f"expected 6 logits, got {logits.size(-1)}")
    probs = F.softmax(logits, dim=-1)
    score_values = torch.arange(
        1, NUM_CLASSES + 1, dtype=probs.dtype, device=probs.device
    )
    return (probs * score_values).sum(dim=-1)


def softmax_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return F.cross_entropy(
        logits, labels, weight=class_weights, label_smoothing=label_smoothing
    )


def combined_expected_score_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    aux_weight: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
    regression: str = "smooth_l1",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """CE on 6 softmax logits + ``aux_weight * reg(E[score], score)``."""
    if aux_weight < 0:
        raise ValueError("aux_weight must be non-negative")
    ce = softmax_ce_loss(
        logits, labels, class_weights=class_weights, label_smoothing=label_smoothing
    )
    expected = expected_score_from_logits(logits)
    target = labels.to(expected.dtype) + 1.0
    if regression == "smooth_l1":
        reg = F.smooth_l1_loss(expected, target, beta=1.0)
    elif regression == "mse":
        reg = F.mse_loss(expected, target)
    elif regression == "l1":
        reg = F.l1_loss(expected, target)
    else:
        raise ValueError(f"unknown regression loss: {regression!r}")
    total = ce + aux_weight * reg
    return total, {
        "ce": float(ce.detach()),
        "expected_reg": float(reg.detach()),
    }


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
class LossComputer:
    """Config-driven loss that also reports its components."""

    def __init__(
        self,
        head_type: str = "softmax",
        objective: str = "ce",
        aux_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        regression: str = "smooth_l1",
        label_smoothing: float = 0.0,
    ):
        self.head_type = head_type
        self.objective = objective
        self.aux_weight = aux_weight
        self.class_weights = class_weights
        self.regression = regression
        self.label_smoothing = label_smoothing
        self._validate()

    def _validate(self) -> None:
        if self.head_type not in {"softmax", "coral"}:
            raise ValueError(f"unknown head_type: {self.head_type!r}")
        valid = {
            ("softmax", "ce"),
            ("softmax", "ce_expected"),
            ("coral", "coral"),
        }
        if (self.head_type, self.objective) not in valid:
            raise ValueError(
                f"inconsistent head_type/objective: ({self.head_type!r}, "
                f"{self.objective!r}); allowed: {sorted(valid)}"
            )

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.objective == "coral":
            loss = coral_loss(logits, labels, class_weights=self.class_weights)
            return loss, {"coral": float(loss.detach())}
        if self.objective == "ce_expected":
            return combined_expected_score_loss(
                logits,
                labels,
                aux_weight=self.aux_weight,
                class_weights=self.class_weights,
                regression=self.regression,
                label_smoothing=self.label_smoothing,
            )
        loss = softmax_ce_loss(
            logits,
            labels,
            class_weights=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        return loss, {"ce": float(loss.detach())}
