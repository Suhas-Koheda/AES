"""Evaluation metrics for Automated Essay Scoring.

Primary metric: Quadratic Weighted Kappa (QWK).  Secondary: MAE, RMSE,
accuracy, Spearman rank correlation.  All metrics are computed on the
original 1..6 score scale; confusion matrices use zero-based class indices.

The QWK implementation is self-contained (no sklearn dependency at import
time) and is cross-checked against ``sklearn.metrics.cohen_kappa_score`` in
the unit tests.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .labels import NUM_CLASSES, class_to_score

# --------------------------------------------------------------------------- #
# Confusion matrix / QWK
# --------------------------------------------------------------------------- #
def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Row = true class, column = predicted class (zero-based indices)."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t < 0 or t >= num_classes or p < 0 or p >= num_classes:
            raise ValueError(f"class out of range: true={t}, pred={p}")
        cm[t, p] += 1
    return cm


def quadratic_weighted_kappa(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> float:
    """Quadratic Weighted Kappa on zero-based class indices.

    Returns ``nan`` when kappa is undefined (e.g. all ratings identical), and
    callers are expected to surface that rather than silently replacing it.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return float("nan")
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return float("nan")

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    idx = np.arange(num_classes)
    weights = (idx[:, None] - idx[None, :]) ** 2 / float((num_classes - 1) ** 2)

    numerator = float((weights * cm).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        # Undefined: expected disagreement is zero (e.g. every rating identical).
        return float("nan")
    return 1.0 - numerator / denominator


# --------------------------------------------------------------------------- #
# Rank correlation
# --------------------------------------------------------------------------- #
def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rho with safe handling of degenerate inputs.

    Returns ``nan`` if either input is constant or has fewer than 2 elements.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size < 2:
        return float("nan")
    rt = _rankdata_average(y_true)
    rp = _rankdata_average(y_pred)
    rt = rt - rt.mean()
    rp = rp - rp.mean()
    denom = np.sqrt((rt**2).sum() * (rp**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((rt * rp).sum() / denom)


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_metrics(
    y_true_scores: Sequence[int],
    y_pred_scores: Sequence[int],
    *,
    y_true_classes: Optional[Sequence[int]] = None,
    y_pred_classes: Optional[Sequence[int]] = None,
    num_classes: int = NUM_CLASSES,
    include_confusion: bool = True,
) -> Dict[str, object]:
    """Compute the full metric suite on the original 1..6 score scale.

    Parameters
    ----------
    y_true_scores, y_pred_scores : original ASAP scores (1..6).
    y_true_classes, y_pred_classes : optional pre-computed class indices.
        If omitted they are derived from the scores.

    Returns a dict with scalar metrics plus (optionally) the confusion matrix.
    Undefined metrics are returned as ``float('nan')`` -- they are never
    silently replaced by 0.
    """
    y_true_scores = np.asarray(y_true_scores, dtype=np.int64).ravel()
    y_pred_scores = np.asarray(y_pred_scores, dtype=np.int64).ravel()
    if y_true_scores.size != y_pred_scores.size:
        raise ValueError("y_true_scores and y_pred_scores must have the same length")
    if y_true_scores.size == 0:
        raise ValueError("cannot compute metrics on an empty set")

    if y_true_classes is None:
        y_true_classes = y_true_scores - 1
    if y_pred_classes is None:
        y_pred_classes = y_pred_scores - 1
    y_true_classes = np.asarray(y_true_classes, dtype=np.int64).ravel()
    y_pred_classes = np.asarray(y_pred_classes, dtype=np.int64).ravel()

    result: Dict[str, object] = {
        "n": int(y_true_scores.size),
        "qwk": quadratic_weighted_kappa(y_true_classes, y_pred_classes, num_classes),
        "mae": _safe_mae(y_true_scores, y_pred_scores),
        "rmse": _safe_rmse(y_true_scores, y_pred_scores),
        "accuracy": float(np.mean(y_true_scores == y_pred_scores)),
        "spearman": spearman_rank_correlation(y_true_scores, y_pred_scores),
        "mean_true_score": float(np.mean(y_true_scores)),
        "mean_pred_score": float(np.mean(y_pred_scores)),
        "std_true_score": float(np.std(y_true_scores)),
        "std_pred_score": float(np.std(y_pred_scores)),
        "exact_agreement_within_1": float(np.mean(np.abs(y_true_scores - y_pred_scores) <= 1)),
    }
    if include_confusion:
        result["confusion_matrix"] = confusion_matrix(
            y_true_classes, y_pred_classes, num_classes=num_classes
        ).tolist()
    return result


def group_metrics(
    y_true_scores: Sequence[int],
    y_pred_scores: Sequence[int],
    groups: Sequence,
) -> Dict[str, Dict[str, object]]:
    """Compute metrics per group (e.g. per prompt, per demographic category)."""
    y_true_scores = np.asarray(y_true_scores, dtype=np.int64).ravel()
    y_pred_scores = np.asarray(y_pred_scores, dtype=np.int64).ravel()
    groups = np.asarray(groups)
    if not (len(groups) == len(y_true_scores) == len(y_pred_scores)):
        raise ValueError("groups, y_true_scores and y_pred_scores must align")

    out: Dict[str, Dict[str, object]] = {}
    for group in sorted(set(groups.tolist()), key=str):
        mask = groups == group
        n = int(mask.sum())
        if n == 0:
            continue
        metrics = compute_metrics(y_true_scores[mask], y_pred_scores[mask])
        metrics.pop("confusion_matrix", None)
        out[str(group)] = metrics
    return out


def aggregate_seed_metrics(per_seed: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Mean/std of scalar metrics across seeds.  NaNs are ignored per metric."""
    if not per_seed:
        return {}
    keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float)) and k != "n"]
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.array([float(run.get(key, np.nan)) for run in per_seed], dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
            "std": float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan"),
            "n_seeds": int(np.isfinite(vals).sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_metrics(metrics: Dict[str, object], prefix: str = "") -> str:
    parts = []
    for key in ("n", "qwk", "mae", "rmse", "accuracy", "spearman"):
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{prefix}{key}={value:.4f}")
            else:
                parts.append(f"{prefix}{key}={value}")
    return " ".join(parts)


def to_plain_floats(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): to_plain_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_floats(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_plain_floats(obj.tolist())
    return obj
