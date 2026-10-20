"""Regression tests for the fairness/error-analysis report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fairness import build_fairness_report


def _preds(n: int = 6) -> dict:
    return {
        "essay_ids": [f"e{i}" for i in range(n)],
        "prompt_names": ["Prompt A", "Prompt A", "Prompt B", "Prompt B", "Prompt C", "Prompt C"],
        "true_score": np.array([1, 2, 3, 4, 5, 6]),
        "predicted_score": np.array([1, 2, 3, 4, 5, 5]),
        "predicted_class": np.array([0, 1, 2, 3, 4, 4]),
        "probabilities": np.full((n, 6), 1.0 / 6.0),
    }


def test_fairness_report_keeps_prompt_name_when_metadata_also_has_it():
    metadata = pd.DataFrame(
        {
            "essay_id": [f"e{i}" for i in range(6)],
            "prompt_name": ["Prompt A", "Prompt A", "Prompt B", "Prompt B", "Prompt C", "Prompt C"],
            "gender": ["F", "M", "F", "M", "F", "M"],
        }
    )

    report = build_fairness_report(_preds(), metadata, ["gender"])

    assert set(report["per_prompt"]) == {"Prompt A", "Prompt B", "Prompt C"}
    assert report["groups"]["gender"]["F"]["n"] == 3
    assert all("prompt_name" in row for row in report["worst_errors"])


def test_fairness_report_groups_missing_values():
    metadata = pd.DataFrame(
        {
            "essay_id": [f"e{i}" for i in range(6)],
            "prompt_name": ["Prompt A", "Prompt A", "Prompt B", "Prompt B", "Prompt C", "Prompt C"],
            "gender": ["F", None, "F", "M", None, "M"],
        }
    )

    report = build_fairness_report(_preds(), metadata, ["gender"])

    assert "__missing__" in report["groups"]["gender"]
    assert report["missing"]["gender"] == 2
