"""Tests for score <-> class conversion and decoding."""

from __future__ import annotations

import numpy as np
import pytest

from src.labels import (
    NUM_CLASSES,
    class_distribution,
    class_to_score,
    coral_class_probabilities,
    decode_coral,
    decode_expected_score,
    decode_logits,
    decode_softmax,
    expected_score_from_probs,
    score_to_class,
    softmax,
)


def test_roundtrip_scalar():
    for score in range(1, 7):
        assert score_to_class(score) == score - 1
        assert class_to_score(score - 1) == score


def test_roundtrip_array():
    scores = np.arange(1, 7)
    classes = score_to_class(scores)
    assert np.array_equal(classes, np.arange(0, 6))
    assert np.array_equal(class_to_score(classes), scores)


def test_roundtrip_list():
    assert [score_to_class(s) for s in [1, 3, 6]] == [0, 2, 5]
    assert [class_to_score(c) for c in [0, 2, 5]] == [1, 3, 6]


@pytest.mark.parametrize("bad", [0, 7, -1, 100])
def test_invalid_score_raises(bad):
    with pytest.raises((ValueError, TypeError)):
        score_to_class(bad)


@pytest.mark.parametrize("bad", [-1, 6, 99])
def test_invalid_class_raises(bad):
    with pytest.raises((ValueError, TypeError)):
        class_to_score(bad)


def test_non_integer_raises():
    with pytest.raises(TypeError):
        score_to_class(3.5)
    with pytest.raises(TypeError):
        score_to_class("3")


def test_softmax_sums_to_one():
    logits = np.random.default_rng(0).normal(size=(20, 6))
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0)


def test_decode_softmax_argmax():
    logits = np.zeros((2, 6))
    logits[0, 2] = 10.0
    logits[1, 5] = 10.0
    classes, scores = decode_softmax(logits)
    assert classes.tolist() == [2, 5]
    assert scores.tolist() == [3, 6]


def test_decode_softmax_rejects_wrong_shape():
    with pytest.raises(ValueError):
        decode_softmax(np.zeros((4, 5)))


def test_expected_score_identity():
    # One-hot-ish distribution centred on class k => expected score k+1.
    probs = np.zeros((3, 6))
    probs[0, 0] = 1.0
    probs[1, 3] = 1.0
    probs[2, 5] = 1.0
    expected = expected_score_from_probs(probs)
    assert np.allclose(expected, [1.0, 4.0, 6.0])


def test_decode_expected_score_rounds_and_clips():
    logits = np.full((1, 6), -100.0)
    logits[0, 4] = 100.0  # class 4 -> score 5
    decoded = decode_expected_score(logits)
    assert decoded.tolist() == [5]


def test_coral_targets_and_decode():
    # Class 3 => sum_k 1[3 > k] = 3 positive cutpoints.
    logits = np.array([[5.0, 5.0, 5.0, -5.0, -5.0]])
    assert decode_coral(logits).tolist() == [3]
    logits_zero = np.zeros((1, 5))
    assert decode_coral(logits_zero).tolist() == [0]  # ties count as not exceeded


def test_coral_class_probabilities_valid_distribution():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 5))
    probs = coral_class_probabilities(logits)
    assert probs.shape == (50, 6)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0)


def test_coral_probs_monotone_case():
    # Strictly decreasing cutpoint logits => a proper unimodal-ish distribution.
    logits = np.array([[6.0, 2.0, -2.0, -6.0, -9.0]])
    probs = coral_class_probabilities(logits)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs.argmax() == 2


def test_decode_logits_dispatch():
    soft = np.zeros((1, 6))
    soft[0, 1] = 5.0
    out = decode_logits(soft, head_type="softmax", mode="argmax")
    assert out["predicted_class"].tolist() == [1]
    assert out["predicted_score"].tolist() == [2]

    coral = np.array([[5.0, 5.0, -5.0, -5.0, -5.0]])
    out = decode_logits(coral, head_type="coral")
    assert out["predicted_class"].tolist() == [2]
    assert out["predicted_score"].tolist() == [3]
    assert out["probabilities"].shape == (1, 6)


def test_class_distribution():
    dist = class_distribution([0, 0, 3, 5])
    assert dist == {0: 2, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1}
