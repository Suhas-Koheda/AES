"""Metric correctness tests, including cross-checks against sklearn/scipy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import (
    aggregate_seed_metrics,
    compute_metrics,
    confusion_matrix,
    group_metrics,
    quadratic_weighted_kappa,
    spearman_rank_correlation,
)


def test_qwk_perfect_agreement():
    y = [0, 1, 2, 3, 4, 5, 2, 3]
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_qwk_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 6, size=500)
    # Imperfect but correlated predictions.
    y_pred = np.clip(y_true + rng.integers(-1, 2, size=500), 0, 5)
    ours = quadratic_weighted_kappa(y_true, y_pred)
    theirs = sklearn.cohen_kappa_score(
        y_true, y_pred, weights="quadratic", labels=list(range(6))
    )
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_qwk_degenerate_returns_nan():
    y = [3, 3, 3, 3]
    assert math.isnan(quadratic_weighted_kappa(y, y))


def test_qwk_bounds():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 6, size=300)
    y_pred = rng.integers(0, 6, size=300)
    value = quadratic_weighted_kappa(y_true, y_pred)
    assert -1.0 <= value <= 1.0


def test_qwk_penalises_large_errors_more():
    y_true = np.array([0, 1, 2, 3, 4, 5] * 10)
    near = np.clip(y_true + 1, 0, 5)
    far = np.clip(y_true + 4, 0, 5)
    assert quadratic_weighted_kappa(y_true, near) > quadratic_weighted_kappa(y_true, far)


def test_confusion_matrix_shape_and_sum():
    cm = confusion_matrix([0, 1, 5], [1, 1, 5])
    assert cm.shape == (6, 6)
    assert cm.sum() == 3
    assert cm[0, 1] == 1 and cm[1, 1] == 1 and cm[5, 5] == 1


def test_confusion_matrix_rejects_out_of_range():
    with pytest.raises(ValueError):
        confusion_matrix([0], [6])


def test_compute_metrics_values():
    y_true = [1, 2, 3, 4, 5, 6]
    y_pred = [1, 2, 3, 4, 5, 5]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["n"] == 6
    assert metrics["mae"] == pytest.approx(1 / 6)
    assert metrics["rmse"] == pytest.approx(math.sqrt(1 / 6))
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["qwk"] == pytest.approx(1.0)
    assert len(metrics["confusion_matrix"]) == 6


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.5, size=200)
    ours = spearman_rank_correlation(a, b)
    theirs = scipy_stats.spearmanr(a, b).statistic
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_spearman_handles_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    a = [1, 2, 2, 3, 4, 4, 4]
    b = [1, 3, 2, 4, 5, 4, 6]
    ours = spearman_rank_correlation(a, b)
    theirs = scipy_stats.spearmanr(a, b).statistic
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_spearman_degenerate_is_nan():
    assert math.isnan(spearman_rank_correlation([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman_rank_correlation([1], [1]))


def test_group_metrics():
    y_true = [1, 2, 3, 4, 1, 2]
    y_pred = [1, 2, 3, 3, 1, 1]
    groups = ["a", "a", "a", "b", "b", "b"]
    result = group_metrics(y_true, y_pred, groups)
    assert set(result) == {"a", "b"}
    assert result["a"]["n"] == 3
    assert result["b"]["n"] == 3


def test_aggregate_seed_metrics_ignores_nan():
    runs = [
        {"qwk": 0.5, "mae": 0.4, "n": 10},
        {"qwk": 0.7, "mae": 0.3, "n": 10},
        {"qwk": float("nan"), "mae": 0.5, "n": 10},
    ]
    agg = aggregate_seed_metrics(runs)
    assert agg["qwk"]["mean"] == pytest.approx(0.6)
    assert agg["qwk"]["n_seeds"] == 2
    assert agg["mae"]["mean"] == pytest.approx(0.4)
