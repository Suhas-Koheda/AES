"""Evaluation: prediction, metrics, per-prompt breakdowns, artefact saving."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .labels import NUM_CLASSES, decode_logits
from .metrics import compute_metrics, group_metrics, to_plain_floats
from .utils import ensure_dir, get_logger, progress, save_json

logger = get_logger()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def move_batch_to_device(batch: dict, device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader,
    device,
    *,
    decode_mode: str = "argmax",
    use_amp: bool = False,
    desc: str = "inference",
    progress_position: int = 0,
) -> dict:
    """Run inference and decode to classes/scores on the original 1..6 scale."""
    model.eval()
    all_logits, all_labels, all_scores = [], [], []
    all_ids, all_prompts, all_n_chunks = [], [], []

    started = time.perf_counter()
    iterator = progress(
        dataloader, desc=desc, unit="batch", position=progress_position
    )
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp and device.type == "cuda"
        ):
            outputs = model(batch)
        logits = outputs["logits"].detach().float().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(batch["labels"].detach().cpu().numpy())
        all_scores.append(batch["scores"].detach().cpu().numpy())
        all_ids.extend(batch["essay_ids"])
        all_prompts.extend(batch["prompt_names"])
        if "n_chunks" in batch:
            all_n_chunks.extend(batch["n_chunks"].detach().cpu().tolist())
        if hasattr(iterator, "set_postfix") and len(all_logits) % 20 == 0:
            iterator.set_postfix(samples=len(all_ids))
    if hasattr(iterator, "close"):
        iterator.close()
    elapsed = time.perf_counter() - started

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    decoded = decode_logits(logits, mode=decode_mode)

    return {
        "logits": logits,
        "true_class": labels,
        "true_score": scores,
        "predicted_class": decoded["predicted_class"],
        "predicted_score": decoded["predicted_score"],
        "probabilities": decoded["probabilities"],
        "essay_ids": all_ids,
        "prompt_names": all_prompts,
        "n_chunks": all_n_chunks,
        "inference_seconds": elapsed,
        "n_samples": int(len(scores)),
    }


# --------------------------------------------------------------------------- #
# Metric + artefact assembly
# --------------------------------------------------------------------------- #
def predictions_to_frame(preds: dict) -> pd.DataFrame:
    probs = preds["probabilities"]
    frame = pd.DataFrame(
        {
            "essay_id": preds["essay_ids"],
            "prompt_name": preds["prompt_names"],
            "true_score": preds["true_score"],
            "predicted_score": preds["predicted_score"],
            "predicted_class": preds["predicted_class"],
            "signed_error": preds["predicted_score"] - preds["true_score"],
            "abs_error": np.abs(preds["predicted_score"] - preds["true_score"]),
        }
    )
    for k in range(probs.shape[1]):
        frame[f"prob_class_{k}"] = probs[:, k]
    if preds.get("n_chunks"):
        frame["n_chunks"] = preds["n_chunks"]
    return frame


def evaluate_predictions(
    preds: dict,
    *,
    metadata: Optional[pd.DataFrame] = None,
    group_columns: Optional[list[str]] = None,
    metric_kwargs: Optional[dict] = None,
) -> dict:
    """Compute overall, per-prompt and optional per-group metrics."""
    metric_kwargs = metric_kwargs or {}
    metrics = compute_metrics(
        preds["true_score"], preds["predicted_score"], **metric_kwargs
    )
    per_prompt = group_metrics(
        preds["true_score"], preds["predicted_score"], preds["prompt_names"],
        **metric_kwargs,
    )

    out = {
        "overall": metrics,
        "per_prompt": per_prompt,
        "inference_seconds": preds["inference_seconds"],
        "n_samples": preds["n_samples"],
        "samples_per_second": (
            preds["n_samples"] / preds["inference_seconds"]
            if preds["inference_seconds"] > 0
            else float("nan")
        ),
    }

    if metadata is not None and group_columns:
        frame = predictions_to_frame(preds)
        merged = frame.merge(metadata, on="essay_id", how="left", suffixes=("", "_meta"))
        group_results = {}
        for column in group_columns:
            if column not in merged.columns:
                logger.warning("group column %r missing from metadata; skipping", column)
                continue
            groups = merged[column].fillna("__missing__").astype(str).to_numpy()
            group_results[column] = group_metrics(
                merged["true_score"].to_numpy(),
                merged["predicted_score"].to_numpy(),
                groups,
                **metric_kwargs,
            )
            # Attach sample sizes explicitly.
            for label, values in group_results[column].items():
                values["n"] = int((groups == label).sum())
        out["by_group"] = group_results

    return to_plain_floats(out)


def save_evaluation(
    preds: dict,
    evaluation: dict,
    out_dir: str | Path,
    split_name: str,
    *,
    metadata: Optional[pd.DataFrame] = None,
) -> Path:
    """Write predictions CSV + metrics JSON; return the predictions path."""
    out_dir = ensure_dir(out_dir)
    frame = predictions_to_frame(preds)
    if metadata is not None:
        merge_cols = ["essay_id"] + [
            c for c in metadata.columns if c != "essay_id" and c not in frame.columns
        ]
        frame = frame.merge(metadata[merge_cols], on="essay_id", how="left")
    pred_path = out_dir / f"predictions_{split_name}.csv"
    frame.to_csv(pred_path, index=False)

    save_json(evaluation, out_dir / f"metrics_{split_name}.json")
    logger.info(
        "Saved %s predictions (%d rows) -> %s", split_name, len(frame), pred_path
    )
    return pred_path


def summarise(evaluation: dict) -> str:
    overall = evaluation["overall"]
    return (
        f"n={overall['n']} QWK={overall['qwk']:.4f} MAE={overall['mae']:.4f} "
        f"RMSE={overall['rmse']:.4f} acc={overall['accuracy']:.4f} "
        f"rho={overall['spearman']:.4f}"
    )


def per_prompt_table(evaluation: dict) -> pd.DataFrame:
    rows = []
    for prompt, values in evaluation.get("per_prompt", {}).items():
        rows.append(
            {
                "prompt_name": prompt,
                "n": values.get("n"),
                "qwk": values.get("qwk"),
                "mae": values.get("mae"),
                "rmse": values.get("rmse"),
                "accuracy": values.get("accuracy"),
                "spearman": values.get("spearman"),
            }
        )
    return pd.DataFrame(rows).sort_values("prompt_name").reset_index(drop=True)


def group_table(evaluation: dict, column: str) -> pd.DataFrame:
    rows = []
    for label, values in evaluation.get("by_group", {}).get(column, {}).items():
        rows.append(
            {
                column: label,
                "n": values.get("n"),
                "qwk": values.get("qwk"),
                "mae": values.get("mae"),
                "rmse": values.get("rmse"),
                "accuracy": values.get("accuracy"),
                "mean_true_score": values.get("mean_true_score"),
                "mean_pred_score": values.get("mean_pred_score"),
            }
        )
    return pd.DataFrame(rows).sort_values(column).reset_index(drop=True)
