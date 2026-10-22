"""Metrics must support ASAP prompts whose native score scales differ."""

import pandas as pd

from src.evaluate import evaluate_predictions
from src.metrics import compute_metrics


def test_asap1_native_qwk_accepts_prompt_one_scores_and_uses_full_scale():
    result = compute_metrics(
        [2, 4, 8, 12], [2, 6, 7, 12], num_classes=11, score_min=2
    )
    assert result["n"] == 4
    assert len(result["confusion_matrix"]) == 11
    assert result["qwk"] <= 1


def test_evaluate_predictions_passes_native_scale_to_prompt_and_group_metrics():
    preds = {
        "true_score": [2, 4, 8, 12],
        "predicted_score": [2, 6, 7, 12],
        "prompt_names": ["1"] * 4,
        "inference_seconds": 1.0,
        "n_samples": 4,
    }
    result = evaluate_predictions(
        preds,
        metric_kwargs={"num_classes": 11, "score_min": 2},
    )
    assert len(result["overall"]["confusion_matrix"]) == 11
    assert result["per_prompt"]["1"]["qwk"] == result["overall"]["qwk"]


def test_evaluate_predictions_joins_numeric_metadata_ids_to_string_prediction_ids():
    preds = {
        "essay_ids": ["101", "102"],
        "prompt_names": ["1", "1"],
        "true_score": [2, 4],
        "predicted_score": [2, 4],
        "predicted_class": [0, 1],
        "probabilities": [[1, 0], [0, 1]],
        "inference_seconds": 1.0,
        "n_samples": 2,
    }
    metadata = pd.DataFrame({"essay_id": [101, 102], "group": ["a", "b"]})
    result = evaluate_predictions(
        preds,
        metadata=metadata,
        group_columns=["group"],
        metric_kwargs={"num_classes": 11, "score_min": 2},
    )
    assert set(result["by_group"]["group"]) == {"a", "b"}
