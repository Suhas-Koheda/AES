"""Metrics must support ASAP prompts whose native score scales differ."""

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
