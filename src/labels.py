"""Label conversion and score decoding for ASAP 2.0.

ASAP 2.0 human holistic scores are integers in ``{1, 2, 3, 4, 5, 6}``.
The neural models use ``numpy``/``torch`` class indices in ``{0, ..., 5}``.

The mapping is intentionally trivial (``class = score - 1``) but every
conversion goes through an explicit, validated function so that raw scores,
class indices, logits, probabilities and decoded scores are never conflated.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

NUM_CLASSES: int = 6
MIN_SCORE: int = 1
MAX_SCORE: int = 6


# --------------------------------------------------------------------------- #
# Score <-> class
# --------------------------------------------------------------------------- #
def score_to_class(score):
    """Map an ASAP score (1..6) to a zero-based class index (0..5).

    Accepts an int, a sequence, or a numpy array. Returns an int for scalar
    input and an ``np.ndarray`` of ``int64`` for array-like input.
    """
    if _is_scalar(score):
        _validate_score(score)
        return int(score) - 1
    arr = np.asarray(score)
    _validate_scores_array(arr)
    return (arr.astype(np.int64) - 1).astype(np.int64)


def class_to_score(class_index):
    """Map a zero-based class index (0..5) back to an ASAP score (1..6)."""
    if _is_scalar(class_index):
        _validate_class(class_index)
        return int(class_index) + 1
    arr = np.asarray(class_index)
    _validate_classes_array(arr)
    return (arr.astype(np.int64) + 1).astype(np.int64)


def _is_scalar(x) -> bool:
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)


def _validate_score(score) -> None:
    if not _is_integer(score):
        raise TypeError(f"score must be an integer in [{MIN_SCORE}, {MAX_SCORE}], got {score!r}")
    if not (MIN_SCORE <= int(score) <= MAX_SCORE):
        raise ValueError(f"score must be in [{MIN_SCORE}, {MAX_SCORE}], got {score!r}")


def _validate_class(index) -> None:
    if not _is_integer(index):
        raise TypeError(f"class index must be an integer in [0, {NUM_CLASSES - 1}], got {index!r}")
    if not (0 <= int(index) <= NUM_CLASSES - 1):
        raise ValueError(f"class index must be in [0, {NUM_CLASSES - 1}], got {index!r}")


def _validate_scores_array(arr: np.ndarray) -> None:
    if arr.size == 0:
        return
    if not np.issubdtype(arr.dtype, np.integer):
        # Allow float arrays that are exactly integral (e.g. from pandas).
        if np.issubdtype(arr.dtype, np.floating) and np.all(arr == np.round(arr)):
            arr = arr.astype(np.int64)
        else:
            raise TypeError(f"score array must contain integers, got dtype {arr.dtype}")
    if arr.min() < MIN_SCORE or arr.max() > MAX_SCORE:
        raise ValueError(
            f"scores must be in [{MIN_SCORE}, {MAX_SCORE}], got range "
            f"[{arr.min()}, {arr.max()}]"
        )


def _validate_classes_array(arr: np.ndarray) -> None:
    if arr.size == 0:
        return
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating) and np.all(arr == np.round(arr)):
            arr = arr.astype(np.int64)
        else:
            raise TypeError(f"class array must contain integers, got dtype {arr.dtype}")
    if arr.min() < 0 or arr.max() > NUM_CLASSES - 1:
        raise ValueError(
            f"class indices must be in [0, {NUM_CLASSES - 1}], got range "
            f"[{arr.min()}, {arr.max()}]"
        )


def _is_integer(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, (float, np.floating)):
        return float(x).is_integer()
    return False


# --------------------------------------------------------------------------- #
# Decoding from model outputs
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def decode_softmax(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``(N, 6)`` logits into ``(predicted_class, predicted_score)``.

    Returns
    -------
    predicted_class : np.ndarray of int, shape (N,)  -- zero-based argmax
    predicted_score : np.ndarray of int, shape (N,)  -- original 1..6 scale
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must have shape (N, {NUM_CLASSES}), got {logits.shape}")
    probs = softmax(logits, axis=-1)
    predicted_class = probs.argmax(axis=-1).astype(np.int64)
    predicted_score = class_to_score(predicted_class)
    return predicted_class, predicted_score


def expected_score_from_probs(probs: np.ndarray) -> np.ndarray:
    """Expected score on the original 1..6 scale from class probabilities.

    ``E[score] = sum_k p_k * (k + 1)``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise ValueError(f"probs must have shape (N, {NUM_CLASSES}), got {probs.shape}")
    score_values = np.arange(MIN_SCORE, MAX_SCORE + 1, dtype=np.float64)
    return probs @ score_values


def decode_expected_score(logits: np.ndarray, rounding: str = "round") -> np.ndarray:
    """Decode logits to the original 1..6 scale via the expected score.

    ``rounding="round"`` uses standard rounding and clips to [1, 6].
    ``rounding="floor"``/``"ceil"`` are provided for completeness.
    """
    probs = softmax(logits, axis=-1)
    expected = expected_score_from_probs(probs)
    if rounding == "round":
        score = np.rint(expected)
    elif rounding == "floor":
        score = np.floor(expected)
    elif rounding == "ceil":
        score = np.ceil(expected)
    else:
        raise ValueError(f"unknown rounding mode: {rounding!r}")
    return np.clip(score, MIN_SCORE, MAX_SCORE).astype(np.int64)


def decode_coral(logits: np.ndarray, thresholds: Sequence[float] | None = None) -> np.ndarray:
    """Decode CORAL ordinal logits to class indices 0..5.

    CORAL uses K-1 = 5 binary logits with a *shared* set of cutpoints.  The
    probability of the score being greater than cutpoint ``k`` is
    ``sigmoid(logit_k)``.  The predicted class is::

        class = sum_k 1[ sigmoid(logit_k) > 0.5 ]

    Since ``sigmoid(x) > 0.5`` iff ``x > 0``, this is just the number of
    positive logits.  ``thresholds`` may override the 0.5 decision boundary
    with per-cutpoint thresholds (e.g. tuned on validation QWK).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES - 1:
        raise ValueError(
            f"CORAL logits must have shape (N, {NUM_CLASSES - 1}), got {logits.shape}"
        )
    if thresholds is None:
        thresholds = np.zeros(NUM_CLASSES - 1)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (NUM_CLASSES - 1,):
        raise ValueError(f"thresholds must have shape ({NUM_CLASSES - 1},)")
    # P(y > k) = sigmoid(logit_k); class = number of cutpoints exceeded.
    class_index = (logits > thresholds[None, :]).sum(axis=-1).astype(np.int64)
    return np.clip(class_index, 0, NUM_CLASSES - 1)


def coral_thresholds_from_logits(logits: np.ndarray) -> np.ndarray:
    """Return the implicit 0.5-probability thresholds (all zeros) for CORAL.

    Present for symmetry/documentation: a zero logit corresponds to p=0.5.
    """
    logits = np.asarray(logits)
    return np.zeros(logits.shape[1], dtype=np.float64)


def coral_class_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert CORAL cutpoint logits into a valid ``(N, 6)`` class distribution.

    ``p(y > k) = sigmoid(logit_k)``.  Monotonicity is enforced with a cumulative
    minimum (documented approximation) so that adjacent differences -- the
    class probabilities -- are non-negative.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES - 1:
        raise ValueError(
            f"CORAL logits must have shape (N, {NUM_CLASSES - 1}), got {logits.shape}"
        )
    p_gt = 1.0 / (1.0 + np.exp(-logits))
    p_gt = np.minimum.accumulate(p_gt, axis=1)
    probs = np.zeros((logits.shape[0], NUM_CLASSES), dtype=np.float64)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for k in range(1, NUM_CLASSES - 1):
        probs[:, k] = p_gt[:, k - 1] - p_gt[:, k]
    probs[:, NUM_CLASSES - 1] = p_gt[:, NUM_CLASSES - 2]
    probs = np.clip(probs, 0.0, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


def decode_logits(
    logits: np.ndarray,
    head_type: str = "softmax",
    mode: str = "argmax",
) -> dict:
    """Single entry point for decoding model logits.

    Parameters
    ----------
    logits : ``(N, 6)`` for a softmax head or ``(N, 5)`` for a CORAL head.
    head_type : ``"softmax"`` or ``"coral"``.
    mode : for softmax, ``"argmax"`` or ``"expected"``; ignored for CORAL.

    Returns a dict with ``predicted_class`` (0..5), ``predicted_score`` (1..6)
    and ``probabilities`` (``(N, 6)``).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if head_type == "softmax":
        probs = softmax(logits, axis=-1)
        if mode == "argmax":
            predicted_class = probs.argmax(axis=-1).astype(np.int64)
        elif mode == "expected":
            expected = expected_score_from_probs(probs)
            predicted_class = np.clip(np.rint(expected), MIN_SCORE, MAX_SCORE).astype(np.int64) - 1
        else:
            raise ValueError(f"unknown softmax decoding mode: {mode!r}")
    elif head_type == "coral":
        probs = coral_class_probabilities(logits)
        predicted_class = decode_coral(logits)
    else:
        raise ValueError(f"unknown head_type: {head_type!r}")

    predicted_score = class_to_score(predicted_class)
    return {
        "predicted_class": predicted_class,
        "predicted_score": predicted_score,
        "probabilities": probs,
    }


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #
def scores_to_classes(scores: Iterable[int]) -> list[int]:
    return [score_to_class(s) for s in scores]


def classes_to_scores(classes: Iterable[int]) -> list[int]:
    return [class_to_score(c) for c in classes]


def class_distribution(classes: Sequence[int]) -> dict[int, int]:
    """Count occurrences of each class index (0..5)."""
    counts = {k: 0 for k in range(NUM_CLASSES)}
    for c in classes:
        counts[int(c)] += 1
    return counts
