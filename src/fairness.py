"""Subgroup fairness and error analysis.

Demographic attributes are used **only** for post-hoc analysis.  They are
never model inputs.  Small groups are reported with their sample size and are
flagged; we make no causal claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .evaluate import _metadata_for_merge, predictions_to_frame
from .metrics import compute_metrics, to_plain_floats
from .utils import ensure_dir, get_logger, save_json

logger = get_logger()

MIN_GROUP_SIZE = 30   # below this we flag, not hide
MIN_GROUP_SIZE_FOR_QWK = 50


def _group_summary(
    frame: pd.DataFrame, column: str, min_size: int
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    values = frame[column].fillna("__missing__").astype(str)
    for label in sorted(values.unique()):
        mask = values == label
        n = int(mask.sum())
        block: dict = {"n": n, "small_sample": n < min_size}
        if n < 2:
            block.update(
                {"qwk": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                 "accuracy": float("nan"), "spearman": float("nan")}
            )
            out[label] = block
            continue
        metrics = compute_metrics(
            frame.loc[mask, "true_score"].to_numpy(),
            frame.loc[mask, "predicted_score"].to_numpy(),
        )
        block.update(
            {
                "qwk": metrics["qwk"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "accuracy": metrics["accuracy"],
                "spearman": metrics["spearman"],
                "mean_true_score": metrics["mean_true_score"],
                "mean_pred_score": metrics["mean_pred_score"],
                "qwk_interpretable": n >= MIN_GROUP_SIZE_FOR_QWK,
            }
        )
        # Score distribution inside the group.
        block["true_distribution"] = (
            frame.loc[mask, "true_score"].value_counts().sort_index().to_dict()
        )
        block["pred_distribution"] = (
            frame.loc[mask, "predicted_score"].value_counts().sort_index().to_dict()
        )
        out[label] = block
    return out


def build_fairness_report(
    preds: dict,
    metadata: pd.DataFrame,
    group_columns: Iterable[str],
    *,
    min_group_size: int = MIN_GROUP_SIZE,
    n_worst: int = 50,
) -> dict:
    """Compute subgroup metrics, score distributions and the worst errors."""
    frame = predictions_to_frame(preds)
    frame["essay_id"] = frame["essay_id"].astype(str)
    metadata = _metadata_for_merge(metadata)
    # ``frame`` already contains ``prompt_name``; merging it again would produce
    # prompt_name_x/prompt_name_y and break the per-prompt summary.
    merge_cols = ["essay_id"] + [
        c for c in metadata.columns if c != "essay_id" and c not in frame.columns
    ]
    if len(merge_cols) > 1:
        frame = frame.merge(metadata[merge_cols], on="essay_id", how="left")

    report: dict = {
        "n": int(len(frame)),
        "groups": {},
        "missing": {},
        "per_prompt": _prompt_summary(frame),
    }

    for column in group_columns:
        if column not in frame.columns:
            report["missing"][column] = "column absent from metadata"
            continue
        report["missing"][column] = int(frame[column].isna().sum())
        report["groups"][column] = _group_summary(frame, column, min_group_size)

    # Worst over- and under-scoring examples.
    error_frame = frame.copy()
    error_frame["abs_error"] = np.abs(
        error_frame["predicted_score"] - error_frame["true_score"]
    )
    cols = [
        "essay_id",
        "prompt_name",
        "true_score",
        "predicted_score",
        "signed_error",
        "abs_error",
    ] + [c for c in group_columns if c in error_frame.columns]
    worst = error_frame.sort_values("abs_error", ascending=False).head(n_worst)
    report["worst_errors"] = worst[cols].to_dict(orient="records")

    over = error_frame[error_frame["signed_error"] > 0].sort_values("signed_error", ascending=False).head(n_worst)
    under = error_frame[error_frame["signed_error"] < 0].sort_values("signed_error").head(n_worst)
    report["largest_over_scores"] = over[cols].to_dict(orient="records")
    report["largest_under_scores"] = under[cols].to_dict(orient="records")

    return to_plain_floats(report)


def _prompt_summary(frame: pd.DataFrame) -> Dict[str, dict]:
    if "prompt_name" not in frame.columns:
        logger.warning("prompt_name missing from predictions/metadata; skipping per-prompt summary")
        return {}
    out = {}
    for prompt in sorted(frame["prompt_name"].unique()):
        mask = frame["prompt_name"] == prompt
        metrics = compute_metrics(
            frame.loc[mask, "true_score"].to_numpy(),
            frame.loc[mask, "predicted_score"].to_numpy(),
        )
        metrics.pop("confusion_matrix", None)
        out[str(prompt)] = to_plain_floats(metrics)
    return out


def save_fairness_report(report: dict, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    save_json(report, out_dir / "fairness_report.json")

    for column, groups in report.get("groups", {}).items():
        rows = []
        for label, values in groups.items():
            row = {"category": label}
            row.update({k: v for k, v in values.items() if not isinstance(v, dict)})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"group_metrics_{column}.csv", index=False)

    for key in ("worst_errors", "largest_over_scores", "largest_under_scores"):
        rows = report.get(key, [])
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"{key}.csv", index=False)

    logger.info("Saved fairness report -> %s", out_dir)


def distribution_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Human vs predicted score distribution per category."""
    rows = []
    for label, group in frame.groupby(frame[column].fillna("__missing__").astype(str)):
        rows.append(
            {
                "category": label,
                "n": len(group),
                **{f"human_{k}": v for k, v in group["true_score"].value_counts().sort_index().items()},
                **{f"pred_{k}": v for k, v in group["predicted_score"].value_counts().sort_index().items()},
            }
        )
    return pd.DataFrame(rows)
