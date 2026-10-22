"""Label mapping and decoding tests for holistic scores 1–6."""

import numpy as np
import pytest

from src.labels import class_to_score, decode_logits, score_to_class, softmax


def test_score_class_roundtrip():
    scores = np.arange(1, 7)
    classes = score_to_class(scores)
    assert classes.tolist() == list(range(6))
    assert class_to_score(classes).tolist() == scores.tolist()


@pytest.mark.parametrize("score", [0, 7, 3.5, "3"])
def test_invalid_scores_rejected(score):
    with pytest.raises((ValueError, TypeError)):
        score_to_class(score)


def test_softmax_is_a_probability_distribution():
    probs = softmax(np.arange(12, dtype=float).reshape(2, 6))
    assert probs.shape == (2, 6)
    assert np.allclose(probs.sum(axis=1), 1)


def test_decode_scores_in_range():
    logits = np.zeros((2, 6))
    logits[0, 2] = 10
    logits[1, 5] = 10
    decoded = decode_logits(logits)
    assert decoded["predicted_class"].tolist() == [2, 5]
    assert decoded["predicted_score"].tolist() == [3, 6]
    assert decoded["probabilities"].shape == (2, 6)


def test_decode_supports_arbitrary_class_count():
    decoded = decode_logits(np.zeros((2, 5)))
    assert decoded["probabilities"].shape == (2, 5)
    assert decoded["predicted_score"].tolist() == [1, 1]
