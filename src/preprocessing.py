"""Text preprocessing and long-input handling.

Design notes
------------
The ASAP 2.0 essays are long (median ~346 words, max ~8,900 words).  We never
silently drop essay endings: the default truncation strategy is ``head_tail``,
which keeps the introduction *and* the conclusion.  ``head`` (standard
right-truncation) is available for comparability, and token-level chunking is
implemented in :mod:`src.collator` for the cross-attention model.

The question and rubric are tokenised under a *separate, smaller* budget so a
long essay can never truncate them to nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .rubric_loader import FIELD_MARKERS

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str, collapse_whitespace: bool = True) -> str:
    """Light, loss-free normalisation.

    We deliberately do **not** lowercase, strip punctuation, or correct
    spelling: mechanics and conventions are part of the holistic rubric.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
@dataclass
class TruncationStats:
    n_total: int = 0
    n_truncated: int = 0
    total_tokens: int = 0
    kept_tokens: int = 0

    def update(self, original_len: int, kept_len: int) -> None:
        self.n_total += 1
        self.total_tokens += original_len
        self.kept_tokens += kept_len
        if kept_len < original_len:
            self.n_truncated += 1

    @property
    def fraction_truncated(self) -> float:
        return self.n_truncated / self.n_total if self.n_total else 0.0

    @property
    def fraction_tokens_kept(self) -> float:
        return self.kept_tokens / self.total_tokens if self.total_tokens else 0.0

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_truncated": self.n_truncated,
            "fraction_truncated": round(self.fraction_truncated, 6),
            "total_tokens": self.total_tokens,
            "kept_tokens": self.kept_tokens,
            "fraction_tokens_kept": round(self.fraction_tokens_kept, 6),
        }


def head_tail_truncate(
    token_ids: Sequence[int],
    max_length: int,
    head_ratio: float = 0.5,
    sep_token_id: int | None = None,
) -> List[int]:
    """Keep the first ``head_ratio`` and the last ``1-head_ratio`` of tokens.

    This preserves both the introduction and the conclusion of an essay.
    An optional separator token is inserted between the head and tail so the
    model can tell that content is missing.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not (0.0 <= head_ratio <= 1.0):
        raise ValueError("head_ratio must be in [0, 1]")

    ids = list(token_ids)
    if len(ids) <= max_length:
        return ids

    if sep_token_id is not None:
        budget = max_length - 1
        if budget <= 0:
            return ids[:max_length]
    else:
        budget = max_length

    head_n = int(round(budget * head_ratio))
    head_n = max(0, min(head_n, budget))
    tail_n = budget - head_n
    head = ids[:head_n]
    tail = ids[len(ids) - tail_n :] if tail_n > 0 else []
    if sep_token_id is not None:
        return head + [sep_token_id] + tail
    return head + tail


def truncate_token_ids(
    token_ids: Sequence[int],
    max_length: int,
    strategy: str = "head_tail",
    head_ratio: float = 0.5,
    sep_token_id: int | None = None,
) -> List[int]:
    """Apply the requested truncation strategy to a list of content ids."""
    ids = list(token_ids)
    if len(ids) <= max_length:
        return ids
    if strategy == "head":
        return ids[:max_length]
    if strategy == "tail":
        return ids[len(ids) - max_length :]
    if strategy == "head_tail":
        return head_tail_truncate(ids, max_length, head_ratio, sep_token_id)
    raise ValueError(f"unknown truncation strategy: {strategy!r}")


# --------------------------------------------------------------------------- #
# Token-length statistics
# --------------------------------------------------------------------------- #
def token_lengths(tokenizer, texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
    """Tokenise without truncation and return content-token counts."""
    lengths: List[int] = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, truncation=False, padding=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(lengths, dtype=np.int64)


def sequence_length_stats(lengths: np.ndarray) -> dict:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return {}
    percentiles = [50, 75, 90, 95, 99, 99.5]
    stats = {
        "n": int(lengths.size),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(lengths, p))
    return stats


def truncation_report(
    lengths: np.ndarray, max_length: int, strategy: str = "head_tail"
) -> dict:
    lengths = np.asarray(lengths, dtype=np.int64)
    stats = TruncationStats()
    for original in lengths:
        kept = min(int(original), max_length)
        stats.update(int(original), kept)
    out = stats.to_dict()
    out.update({"max_length": max_length, "strategy": strategy})
    return out


# --------------------------------------------------------------------------- #
# Input assembly
# --------------------------------------------------------------------------- #
def build_context_text(question: str, rubric: str) -> str:
    """Question and rubric text used as cross-attention key/value memory."""
    return (
        f"{FIELD_MARKERS['question']} {question.strip()}\n"
        f"{FIELD_MARKERS['rubric']} {rubric.strip()}"
    )
