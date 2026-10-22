"""ASAP 1.0 adapter and prompt-wise five-fold protocol.

The model retains its six-class head. ASAP 1.0 native score ranges vary by
prompt, so the published native range is mapped to six ordinal bins for
training, then decoded back to the native range for reported metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import assert_no_leakage

PROMPT_SCORE_RANGES = {
    1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
    5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60),
}


def load_asap1(data_path: str | Path, contexts_path: str | Path) -> pd.DataFrame:
    """Read a labeled ASAP 1.0 CSV/TSV and attach genuine supplied context.

    ``contexts_path`` must be JSON mapping string prompt IDs 1..8 to the
    prompt/rubric text transcribed from the source materials. Empty/generated
    context is rejected; no rubric text is synthesized by this adapter.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"ASAP 1.0 labeled data not found: {path}")
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    # The original Kaggle TSV is Latin-1 encoded despite the essays being
    # distributed as text. Latin-1 is also a lossless byte mapping for ASCII.
    df = pd.read_csv(path, sep=sep, encoding="latin-1")
    aliases = {
        "essay_set": "prompt_id", "essay": "full_text", "domain1_score": "score",
    }
    for source, target in aliases.items():
        if target not in df and source in df:
            df[target] = df[source]
    required = {"prompt_id", "full_text", "score"}
    if missing := required - set(df.columns):
        raise ValueError(f"ASAP 1.0 data missing columns: {sorted(missing)}")
    df["prompt_id"] = pd.to_numeric(df["prompt_id"], errors="raise").astype(int)
    if not df["prompt_id"].isin(PROMPT_SCORE_RANGES).all():
        raise ValueError("ASAP 1.0 prompt IDs must be integers 1..8")
    if df["full_text"].isna().any() or (df["full_text"].astype(str).str.strip() == "").any():
        raise ValueError("ASAP 1.0 essays must have non-empty text")
    df["score"] = pd.to_numeric(df["score"], errors="raise")
    if df["score"].isna().any() or not np.equal(df["score"], np.floor(df["score"])).all():
        raise ValueError("ASAP 1.0 labels must be non-null integers")
    for prompt, (low, high) in PROMPT_SCORE_RANGES.items():
        scores = df.loc[df.prompt_id == prompt, "score"]
        if not scores.empty and not scores.between(low, high).all():
            raise ValueError(f"prompt {prompt} score outside published range {low}..{high}")
    if "essay_id" not in df:
        df["essay_id"] = [f"{p}:{i}" for p, i in zip(df.prompt_id, df.index)]
    if df.essay_id.isna().any() or df.essay_id.duplicated().any():
        raise ValueError("ASAP 1.0 essay IDs must be non-null and unique")

    contexts_path = Path(contexts_path)
    if not contexts_path.is_file():
        raise FileNotFoundError(
            f"ASAP 1.0 prompt context file missing: {contexts_path}. "
            "Supply source prompt/rubric text; this adapter does not invent it."
        )
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    missing_contexts = [str(i) for i in PROMPT_SCORE_RANGES if not str(contexts.get(str(i), "")).strip()]
    if missing_contexts:
        raise ValueError(f"context file needs non-empty source text for prompts {missing_contexts}")

    df["prompt_name"] = df.prompt_id.astype(str)
    df["assignment"] = df.prompt_id.map(lambda p: str(contexts[str(p)]))
    df["raw_score"] = df.score.astype(int)
    # Known published prompt score bounds, fixed independently of held-out rows.
    labels = []
    for prompt, score in zip(df.prompt_id, df.raw_score):
        low, high = PROMPT_SCORE_RANGES[int(prompt)]
        labels.append(int(np.rint((int(score) - low) * 5 / (high - low))))
    df["label"] = np.asarray(labels, dtype=np.int64)
    df["score"] = df.label + 1  # model's unchanged six-class scale
    df["full_text"] = df.full_text.astype(str)
    return df.reset_index(drop=True)


def restore_native_scores(prompt_id: int, class_indices) -> np.ndarray:
    """Inverse six-bin mapping, rounded to the original prompt's score scale."""
    low, high = PROMPT_SCORE_RANGES[int(prompt_id)]
    classes = np.asarray(class_indices, dtype=np.float64)
    return np.rint(low + classes * (high - low) / 5).astype(np.int64)


def make_prompt_fold(df: pd.DataFrame, prompt_id: int, fold: int, seed: int):
    """Create the paper's prompt-specific 60/20/20 five-fold partition.

    Within each observed native score category, shuffle deterministically and
    distribute records round-robin among five folds. Fold ``i`` is test,
    ``(i+1) mod 5`` is validation, and the remaining three folds train.
    """
    if fold not in range(5):
        raise ValueError("fold must be in 0..4")
    prompt = df.loc[df.prompt_id == prompt_id].reset_index(drop=True)
    if prompt.empty:
        raise ValueError(f"no ASAP 1.0 records for prompt {prompt_id}")
    rng = np.random.default_rng(seed)
    fold_ids = [[] for _ in range(5)]
    for score in sorted(prompt.raw_score.unique()):
        indexes = np.flatnonzero(prompt.raw_score.to_numpy() == score)
        rng.shuffle(indexes)
        for i, index in enumerate(indexes):
            fold_ids[i % 5].append(int(index))
    test_fold, val_fold = fold, (fold + 1) % 5
    train_idx = [i for f, ids in enumerate(fold_ids) if f not in (test_fold, val_fold) for i in ids]
    train = prompt.iloc[train_idx].reset_index(drop=True)
    val = prompt.iloc[fold_ids[val_fold]].reset_index(drop=True)
    test = prompt.iloc[fold_ids[test_fold]].reset_index(drop=True)
    if min(map(len, (train, val, test))) == 0:
        raise ValueError(f"prompt {prompt_id} lacks records for five-fold 60/20/20 splitting")
    assert_no_leakage(train, val, test)
    return train, val, test
