"""Loss-function correctness tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.labels import NUM_CLASSES
from src.losses import (
    LossComputer,
    combined_expected_score_loss,
    compute_class_weights,
    coral_loss,
    coral_targets,
    expected_score_from_logits,
    softmax_ce_loss,
)


def test_coral_targets_are_monotone():
    labels = torch.tensor([0, 2, 5])
    targets = coral_targets(labels)
    assert targets.shape == (3, NUM_CLASSES - 1)
    assert targets[0].tolist() == [0, 0, 0, 0, 0]
    assert targets[1].tolist() == [1, 1, 0, 0, 0]
    assert targets[2].tolist() == [1, 1, 1, 1, 1]


def test_coral_loss_matches_manual_bce():
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 3, 5])
    ours = coral_loss(logits, labels)
    manual = F.binary_cross_entropy_with_logits(logits, coral_targets(labels))
    assert torch.allclose(ours, manual)


def test_coral_loss_prefers_correct_ordering():
    labels = torch.tensor([4])
    good = torch.tensor([[5.0, 5.0, 5.0, 5.0, -5.0]])
    bad = torch.tensor([[-5.0, -5.0, -5.0, -5.0, 5.0]])
    assert coral_loss(good, labels) < coral_loss(bad, labels)


def test_coral_loss_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        coral_loss(torch.randn(2, 6), torch.tensor([1, 2]))


def test_expected_score_from_logits_one_hot():
    logits = torch.tensor([[100.0, -100, -100, -100, -100, -100]])
    assert torch.allclose(expected_score_from_logits(logits), torch.tensor([1.0]))


def test_combined_loss_decomposition():
    torch.manual_seed(0)
    logits = torch.randn(8, 6)
    labels = torch.randint(0, 6, (8,))
    total, parts = combined_expected_score_loss(logits, labels, aux_weight=0.7)
    ce = softmax_ce_loss(logits, labels)
    expected = expected_score_from_logits(logits)
    reg = F.smooth_l1_loss(expected, labels.float() + 1.0, beta=1.0)
    assert torch.allclose(total, ce + 0.7 * reg)
    assert parts["ce"] == pytest.approx(float(ce))


def test_combined_loss_zero_weight_equals_ce():
    torch.manual_seed(0)
    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 1, 2, 3])
    total, _ = combined_expected_score_loss(logits, labels, aux_weight=0.0)
    assert torch.allclose(total, softmax_ce_loss(logits, labels))


def test_class_weights_balanced():
    weights = compute_class_weights([10, 10, 20, 20, 40, 40], NUM_CLASSES, "balanced")
    assert weights is not None
    assert weights.shape == (6,)
    assert weights[0] > weights[5]  # rarer class -> larger weight
    assert compute_class_weights([1] * 6, NUM_CLASSES, "none") is None


def test_loss_computer_rejects_inconsistent_combo():
    with pytest.raises(ValueError):
        LossComputer(head_type="coral", objective="ce")
    with pytest.raises(ValueError):
        LossComputer(head_type="softmax", objective="coral")


def test_loss_computer_dispatch():
    labels = torch.tensor([0, 3, 5])
    softmax_logits = torch.randn(3, 6)
    coral_logits = torch.randn(3, 5)
    ce_loss, ce_parts = LossComputer("softmax", "ce")(softmax_logits, labels)
    assert "ce" in ce_parts and torch.isfinite(ce_loss)
    coral_l, coral_parts = LossComputer("coral", "coral")(coral_logits, labels)
    assert "coral" in coral_parts and torch.isfinite(coral_l)
    combined, parts = LossComputer("softmax", "ce_expected", aux_weight=0.5)(
        softmax_logits, labels
    )
    assert torch.isfinite(combined)
    assert set(parts) == {"ce", "expected_reg"}
