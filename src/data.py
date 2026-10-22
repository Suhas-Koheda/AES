"""Dataset loading, schema verification, splits and leakage checks.

Official ASAP 2.0 layout (verified from the released files):

- ``ASAP_2_Final_github_train.csv`` : 17,307 essays, 7 prompts, ``set=train``,
  ``pubpriv=0``.  This is the *official training partition*.
- ``ASAP_2_Final_github_test.csv``  : 7,421 essays, **5 prompts**
  (``Car-free cities`` and ``Does the electoral college work?`` are absent
  from the released test file), ``set=test``, ``pubpriv`` in
  {``public``, ``private``}.

There is **no official validation partition**.  Evaluation 1 therefore carves a
stratified validation set out of the official training partition (never out of
test) and keeps the official test file untouched until final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler

from .preprocessing import clean_text, word_count
from .utils import get_logger

logger = get_logger()

CORE_COLUMNS = [
    "essay_id",
    "score",
    "full_text",
    "set",
    "pubpriv",
    "assignment",
    "prompt_name",
]

DEMOGRAPHIC_COLUMNS = [
    "economically_disadvantaged",
    "student_disability_status",
    "ell_status",
    "race_ethnicity",
    "gender",
    "grade_level",
]

EXPECTED_TEST_SIZE = 7421

# --------------------------------------------------------------------------- #
# Schema verification
# --------------------------------------------------------------------------- #
def fill_missing_assignments(
    df: pd.DataFrame, name: str = "data"
) -> tuple[pd.DataFrame, int]:
    """Fill null ``assignment`` values from the prompt-level assignment text.

    The released ASAP 2.0 files contain a small number of rows (18 train,
    10 test as of the 2025 release) whose ``assignment`` is empty, even though
    every prompt has exactly one assignment text elsewhere in the same file.
    Because ``assignment`` is a *prompt-level* attribute, the correct repair is
    to copy the prompt's text into those rows. Nothing is dropped and no
    example-level information is invented.

    Raises if a prompt with missing assignments has no non-null example.
    """
    missing = df["assignment"].isna()
    if not missing.any():
        return df, 0

    if df.loc[missing, "prompt_name"].isna().any():
        raise ValueError(
            f"[{name}] cannot fill missing assignment for rows with null prompt_name"
        )

    available = df.loc[~missing]
    mapping = available.groupby("prompt_name")["assignment"].first()

    distinct = available.groupby("prompt_name")["assignment"].nunique()
    ambiguous = distinct[distinct > 1]
    if len(ambiguous):
        logger.warning(
            "[%s] prompts with more than one distinct assignment text: %s; "
            "using the first for back-filling",
            name,
            ambiguous.to_dict(),
        )

    unresolved = sorted(set(df.loc[missing, "prompt_name"]) - set(mapping.index))
    if unresolved:
        raise ValueError(
            f"[{name}] {int(missing.sum())} rows have a null assignment and "
            f"prompts {unresolved} have no non-null example to copy from"
        )

    df = df.copy()
    df.loc[missing, "assignment"] = df.loc[missing, "prompt_name"].map(mapping)
    filled = int(missing.sum())
    logger.warning(
        "[%s] filled %d null assignment value(s) from the prompt-level text "
        "(%s)",
        name,
        filled,
        sorted(set(df.loc[missing, "prompt_name"])),
    )
    return df, filled


def verify_schema(df: pd.DataFrame, name: str, require_score: bool = True) -> dict:
    """Validate the ASAP schema and return a compact summary.

    ``assignment`` may be null only if :func:`fill_missing_assignments` has not
    been applied yet; this function reports the count so callers can decide.
    Raises ``ValueError`` on structural violations.
    """
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"[{name}] dataframe is empty")

    if df["essay_id"].isna().any():
        raise ValueError(f"[{name}] contains null essay_id values")
    if df["essay_id"].duplicated().any():
        n_dup = int(df["essay_id"].duplicated().sum())
        raise ValueError(f"[{name}] contains {n_dup} duplicated essay_id values")
    if df["full_text"].isna().any():
        raise ValueError(f"[{name}] contains null full_text values")
    if df["prompt_name"].isna().any():
        raise ValueError(f"[{name}] contains null prompt_name values")
    if df["assignment"].isna().any():
        affected = sorted(df.loc[df["assignment"].isna(), "prompt_name"].unique())
        raise ValueError(
            f"[{name}] contains {int(df['assignment'].isna().sum())} null "
            f"assignment values (prompts: {affected}). Call "
            "`fill_missing_assignments` first, or use `load_asap_data`, which "
            "fills them from the prompt-level text."
        )

    if require_score:
        if df["score"].isna().any():
            raise ValueError(f"[{name}] contains null scores")
        if not df["score"].between(1, 6).all():
            bad = sorted(set(df.loc[~df["score"].between(1, 6), "score"].tolist()))
            raise ValueError(f"[{name}] scores outside 1..6: {bad}")

    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_name"].nunique()),
        "prompts": df["prompt_name"].value_counts().to_dict(),
        "score_counts": df["score"].value_counts().sort_index().to_dict(),
        "n_missing_full_text": int(df["full_text"].isna().sum()),
        "n_duplicate_essay_ids": int(df["essay_id"].duplicated().sum()),
        "columns": list(df.columns),
    }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_asap_data(
    train_csv: str | Path,
    test_csv: Optional[str | Path] = None,
    *,
    clean: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load the official train/test CSVs, validate them, and add helper columns."""
    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(
            f"official training CSV not found: {train_csv}. "
            "Run `python scripts/prepare_data.py --repo-dir data/asap2_raw` first."
        )
    train_df = pd.read_csv(train_csv)
    train_df, n_filled_train = fill_missing_assignments(train_df, "train")
    verify_schema(train_df, "train")

    test_df = None
    n_filled_test = 0
    if test_csv is not None:
        test_csv = Path(test_csv)
        if not test_csv.exists():
            raise FileNotFoundError(
                f"official test CSV not found: {test_csv}. "
                "The official test archive is password-protected (password: asap2_test)."
            )
        test_df = pd.read_csv(test_csv)
        test_df, n_filled_test = fill_missing_assignments(test_df, "test")
        verify_schema(test_df, "test")

    for df in (train_df, test_df):
        if df is None:
            continue
        if clean:
            df["full_text"] = df["full_text"].map(clean_text)
            df["assignment"] = df["assignment"].map(clean_text)
        df["label"] = (df["score"].astype(int) - 1).astype(int)
        df["essay_word_count_recomputed"] = df["full_text"].map(word_count)

    train_df.attrs["n_filled_assignment"] = n_filled_train
    if test_df is not None:
        test_df.attrs["n_filled_assignment"] = n_filled_test

    return train_df, test_df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class SplitManifest:
    protocol: str
    seed: int
    val_fraction: float
    train_ids: List[str] = field(default_factory=list)
    val_ids: List[str] = field(default_factory=list)
    test_ids: List[str] = field(default_factory=list)
    heldout_prompt: Optional[str] = None

    def summary(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        def _dist(df: pd.DataFrame) -> dict:
            return {
                "n": int(len(df)),
                "prompts": df["prompt_name"].value_counts().to_dict(),
                "scores": df["score"].value_counts().sort_index().to_dict(),
            }

        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "heldout_prompt": self.heldout_prompt,
            "train": _dist(train_df),
            "val": _dist(val_df),
            "test": _dist(test_df),
        }


def _stratified_split_indices(
    keys: Sequence, val_fraction: float, seed: int
) -> tuple[List[int], List[int]]:
    """Deterministic stratified split ensuring >=1 training row per stratum."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)

    groups: Dict[object, List[int]] = {}
    for idx, key in enumerate(keys):
        groups.setdefault(key, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for key in sorted(groups, key=str):
        idxs = list(groups[key])
        rng.shuffle(idxs)
        n = len(idxs)
        if n < 2:
            train_idx.extend(idxs)
            continue
        n_val = int(round(n * val_fraction))
        n_val = max(1, n_val)
        n_val = min(n_val, n - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_train_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
    stratum_cols: Sequence[str] = ("prompt_name", "score"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify jointly on prompt and score so every prompt keeps its score spread."""
    for col in stratum_cols:
        if col not in df.columns:
            raise ValueError(f"missing stratum column: {col}")
    keys = list(zip(*[df[col].astype(str).tolist() for col in stratum_cols]))
    train_idx, val_idx = _stratified_split_indices(keys, val_fraction, seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def make_official_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Evaluation 1: official train/val/test.

    The validation set is carved from the official training partition because
    ASAP 2.0 releases no validation split.  The official test partition is
    returned untouched.
    """
    train_split, val_split = stratified_train_val_split(train_df, val_fraction, seed)
    manifest = SplitManifest(
        protocol="official",
        seed=seed,
        val_fraction=val_fraction,
        train_ids=train_split["essay_id"].tolist(),
        val_ids=val_split["essay_id"].tolist(),
        test_ids=test_df["essay_id"].tolist(),
    )
    assert_no_leakage(train_split, val_split, test_df)
    return train_split, val_split, test_df, manifest


def make_heldout_prompt_splits(
    df: pd.DataFrame,
    heldout_prompt: str,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Evaluation 2: one prompt is excluded from train *and* validation.

    The held-out prompt's essays are used only for testing.  The remaining
    prompts are split into train/val for model selection.
    """
    prompts = sorted(df["prompt_name"].unique().tolist())
    if heldout_prompt not in prompts:
        raise ValueError(f"heldout prompt {heldout_prompt!r} not in {prompts}")

    heldout_mask = df["prompt_name"] == heldout_prompt
    test_split = df.loc[heldout_mask].reset_index(drop=True)
    remaining = df.loc[~heldout_mask].reset_index(drop=True)
    if len(remaining) == 0:
        raise ValueError("no training data left after holding out a prompt")

    train_split, val_split = stratified_train_val_split(remaining, val_fraction, seed)

    manifest = SplitManifest(
        protocol="heldout_prompt",
        seed=seed,
        val_fraction=val_fraction,
        train_ids=train_split["essay_id"].tolist(),
        val_ids=val_split["essay_id"].tolist(),
        test_ids=test_split["essay_id"].tolist(),
        heldout_prompt=heldout_prompt,
    )
    assert_no_leakage(train_split, val_split, test_split)
    # Extra guarantee: the held-out prompt must be absent from train/val.
    assert heldout_prompt not in set(train_split["prompt_name"])
    assert heldout_prompt not in set(val_split["prompt_name"])
    return train_split, val_split, test_split, manifest


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #
def assert_no_leakage(*dfs: pd.DataFrame) -> None:
    """Essay IDs must never appear in more than one split."""
    seen: Dict[str, str] = {}
    names = ["split_0", "split_1", "split_2", "split_3"]
    for name, df in zip(names, dfs):
        for essay_id in df["essay_id"].astype(str):
            if essay_id in seen:
                raise AssertionError(
                    f"data leakage: essay_id {essay_id!r} appears in both "
                    f"{seen[essay_id]} and {name}"
                )
            seen[essay_id] = name


def check_prompt_disjointness(train_df: pd.DataFrame, test_df: pd.DataFrame, heldout_prompt: str) -> None:
    """Raise if the held-out prompt leaks into training."""
    if heldout_prompt in set(train_df["prompt_name"]):
        raise AssertionError(f"held-out prompt {heldout_prompt!r} appears in training data")


def save_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest: SplitManifest,
    out_dir: str | Path,
) -> None:
    """Write a per-essay split manifest CSV and a JSON summary."""
    from .utils import ensure_dir, save_json

    out_dir = ensure_dir(out_dir)
    rows = []
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        for essay_id, prompt, score in zip(df["essay_id"], df["prompt_name"], df["score"]):
            rows.append(
                {
                    "essay_id": essay_id,
                    "split": split_name,
                    "prompt_name": prompt,
                    "score": int(score),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "split_manifest.csv", index=False)
    save_json(manifest.summary(train_df, val_df, test_df), out_dir / "split_summary.json")


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
class EssayDataset(Dataset):
    """Returns raw fields; tokenisation happens in the collator (dynamic padding)."""

    def __init__(
        self,
        df: pd.DataFrame,
        rubric_text: str,
    ):
        required = {"essay_id", "full_text", "assignment", "prompt_name", "score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"EssayDataset missing columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True)
        self.rubric_text = rubric_text

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        prompt = str(row["prompt_name"])
        return {
            "essay_id": str(row["essay_id"]),
            "question": str(row["assignment"]),
            "answer": str(row["full_text"]),
            "rubric": self.rubric_text,
            "label": int(row["label"]) if "label" in row.index else int(row["score"]) - 1,
            "score": int(row["score"]),
            "prompt_name": prompt,
        }

    def metadata_frame(self) -> pd.DataFrame:
        cols = [c for c in ["essay_id", "prompt_name", "score"] + DEMOGRAPHIC_COLUMNS if c in self.df.columns]
        return self.df[cols].copy()


class PromptGroupedBatchSampler(Sampler[list[int]]):
    """Batch indices so that most batches contain a single prompt.

    The question+rubric context is identical for every essay of a prompt.
    Contiguous per-prompt batches let :class:`RubricCrossAttentionModel`
    encode that context once per batch instead of once per essay, which is the
    main memory/compute cost of the two-pass design.  Indices are shuffled
    within prompts and the resulting batches are shuffled each epoch, so
    nothing about the data order is fixed.
    """

    def __init__(
        self,
        prompt_names: Sequence[str],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self._epoch = 0
        self.groups: Dict[str, List[int]] = {}
        for index, prompt in enumerate(prompt_names):
            self.groups.setdefault(str(prompt), []).append(index)
        self._prompts = sorted(self.groups)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        prompts = list(self._prompts)
        if self.shuffle:
            rng.shuffle(prompts)

        order: List[int] = []
        for prompt in prompts:
            indices = np.asarray(self.groups[prompt], dtype=np.int64)
            if self.shuffle:
                indices = rng.permutation(indices)
            order.extend(int(i) for i in indices)

        batches = [
            order[start : start + self.batch_size]
            for start in range(0, len(order), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        n = sum(len(v) for v in self.groups.values())
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
