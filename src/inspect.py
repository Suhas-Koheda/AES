"""Dataset and rubric inspection utilities.

Produces a machine-readable report plus human-readable console output covering:
schema, split values, prompt/score distributions, demographic coverage,
essay word counts, token-length quantiles, truncation counts at candidate
maximum lengths, and the parsed rubric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .data import DEMOGRAPHIC_COLUMNS, load_asap_data, verify_schema
from .preprocessing import sequence_length_stats, token_lengths, truncation_report
from .rubric_loader import load_rubric
from .utils import get_logger, save_json

logger = get_logger()


def _value_counts(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {"__missing_column__": True}
    return {
        str(k): int(v)
        for k, v in df[column].value_counts(dropna=False).sort_index().items()
    }


def _describe_split(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_name"].nunique()),
        "n_filled_assignment": int(df.attrs.get("n_filled_assignment", 0)),
        "set_values": _value_counts(df, "set"),
        "pubpriv_values": _value_counts(df, "pubpriv"),
        "task_values": _value_counts(df, "task"),
        "score_counts": _value_counts(df, "score"),
        "score_percent": {
            str(k): round(100 * v / len(df), 3)
            for k, v in df["score"].value_counts().sort_index().items()
        },
        "prompt_counts": _value_counts(df, "prompt_name"),
        "score_by_prompt": pd.crosstab(df["prompt_name"], df["score"]).to_dict(),
        "grade_level": _value_counts(df, "grade_level"),
        "demographics": {
            column: _value_counts(df, column) for column in DEMOGRAPHIC_COLUMNS
        },
        "essay_word_count": {
            "mean": float(df["essay_word_count"].mean()),
            "median": float(df["essay_word_count"].median()),
            "min": float(df["essay_word_count"].min()),
            "max": float(df["essay_word_count"].max()),
        },
        "assignment_unique_per_prompt": {
            str(prompt): int(group["assignment"].nunique())
            for prompt, group in df.groupby("prompt_name")
        },
    }


def inspect_dataset(
    train_csv: str | Path,
    test_csv: Optional[str | Path] = None,
    rubric_path: Optional[str | Path] = None,
    *,
    tokenizer_name: Optional[str] = None,
    max_lengths: Sequence[int] = (256, 512, 768, 1024),
    out_path: Optional[str | Path] = None,
    sample_for_tokenizer: Optional[int] = None,
) -> dict:
    """Inspect the official files and optionally token-length distributions."""
    train_df, test_df = load_asap_data(train_csv, test_csv)
    report: Dict[str, object] = {
        "train": _describe_split(train_df, "train"),
    }
    if test_df is not None:
        report["test"] = _describe_split(test_df, "test")
        report["overlap"] = {
            "essay_id_overlap": int(len(set(train_df["essay_id"]) & set(test_df["essay_id"]))),
            "full_text_overlap": int(len(set(train_df["full_text"]) & set(test_df["full_text"]))),
        }
        report["prompt_coverage"] = {
            "train_only": sorted(set(train_df["prompt_name"]) - set(test_df["prompt_name"])),
            "test_only": sorted(set(test_df["prompt_name"]) - set(train_df["prompt_name"])),
            "shared": sorted(set(train_df["prompt_name"]) & set(test_df["prompt_name"])),
        }

    if rubric_path is not None:
        rubric = load_rubric(rubric_path)
        report["rubric"] = {
            **rubric.describe(),
            "descriptions": rubric.descriptions,
            "full_text": rubric.full_text,
        }

    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        texts = train_df["full_text"].astype(str).tolist()
        if sample_for_tokenizer and sample_for_tokenizer < len(texts):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(texts), size=sample_for_tokenizer, replace=False)
            texts = [texts[i] for i in idx]
        lengths = token_lengths(tokenizer, texts)
        report["token_lengths"] = {
            "tokenizer": tokenizer_name,
            "sample_size": len(texts),
            "stats": sequence_length_stats(lengths),
            "truncation": {
                str(max_len): truncation_report(lengths, max_len, "head_tail")
                for max_len in max_lengths
            },
        }

    _print_report(report)
    if out_path:
        save_json(report, out_path)
        logger.info("Saved inspection report -> %s", out_path)
    return report


def _print_report(report: dict) -> None:
    for split in ("train", "test"):
        if split not in report:
            continue
        block = report[split]
        logger.info("=" * 70)
        logger.info("%s: %d rows, %d prompts", split.upper(), block["n_rows"], block["n_prompts"])
        logger.info("  prompts: %s", block["prompt_counts"])
        logger.info("  scores : %s", block["score_counts"])
        logger.info("  pubpriv: %s", block["pubpriv_values"])
        logger.info("  word count: %s", block["essay_word_count"])
        if block.get("n_filled_assignment"):
            logger.info(
                "  filled %d null assignment value(s) from prompt-level text",
                block["n_filled_assignment"],
            )
        for column, counts in block["demographics"].items():
            logger.info("  %s: %s", column, counts)

    if "prompt_coverage" in report:
        logger.info("Prompt coverage: %s", report["prompt_coverage"])
    if "rubric" in report:
        rubric = report["rubric"]
        logger.info(
            "Rubric: %s bands=%s words=%d shared_across_prompts=%s",
            rubric["source"],
            rubric["score_bands"],
            rubric["num_words"],
            rubric["shared_across_prompts"],
        )
    if "token_lengths" in report:
        stats = report["token_lengths"]["stats"]
        logger.info("Token lengths (%s): %s", report["token_lengths"]["tokenizer"], stats)
        for max_len, trunc in report["token_lengths"]["truncation"].items():
            logger.info(
                "  max_length=%s: %.2f%% essays truncated, %.2f%% tokens kept",
                max_len,
                100 * trunc["fraction_truncated"],
                100 * trunc["fraction_tokens_kept"],
            )
