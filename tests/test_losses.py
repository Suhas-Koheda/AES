"""Cross-entropy and label-weighting checks."""

import pytest
import torch
import torch.nn.functional as F

from src.losses import LossComputer, compute_class_weights


def test_cross_entropy_matches_torch():
    logits = torch.randn(8, 6, generator=torch.Generator().manual_seed(4))
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 2, 4])
    loss, parts = LossComputer()(logits, labels)
    assert torch.allclose(loss, F.cross_entropy(logits, labels))
    assert set(parts) == {"ce"}


def test_balanced_class_weights():
    weights = compute_class_weights([10, 5, 0, 5, 10, 20], mode="balanced")
    assert weights.shape == (6,)
    assert torch.isfinite(weights).all()


def test_class_weight_mode_validation():
    with pytest.raises(ValueError):
        compute_class_weights([1] * 6, mode="unknown")


def test_loss_rejects_wrong_head_shape():
    with pytest.raises(ValueError):
        LossComputer()(torch.randn(3, 5), torch.tensor([0, 1, 2]))
