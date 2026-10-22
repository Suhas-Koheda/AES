"""Dynamic padding and chunking for essay/question/rubric cross-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .preprocessing import build_context_text, truncate_token_ids



@dataclass
class CollatorConfig:
    input_mode: str = "qra_cross"
    max_answer_length: int = 512
    max_context_length: int = 512
    truncation: str = "head_tail"
    head_ratio: float = 0.5
    chunk: bool = False
    chunk_size: int | None = None
    chunk_overlap: int = 32
    max_chunks: int = 4

    def validate(self) -> None:
        if self.input_mode != "qra_cross":
            raise ValueError("only input_mode='qra_cross' is supported")
        if self.max_answer_length <= 0:
            raise ValueError("max_answer_length must be positive")
        if self.max_context_length < 0:
            raise ValueError("max_context_length must be non-negative")
        if self.chunk:
            if not self.chunk_size or self.chunk_size <= 0:
                raise ValueError("chunk=True requires a positive chunk_size")
            if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be in [0, chunk_size)")
            if self.max_chunks < 1:
                raise ValueError("max_chunks must be >= 1")


def make_windows(
    token_ids: Sequence[int],
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> List[List[int]]:
    """Split ids into overlapping windows, capping the count.

    When capping, the **first** and **last** windows are always kept so the
    essay introduction and conclusion both survive.
    """
    ids = list(token_ids)
    if len(ids) <= chunk_size:
        return [ids]
    step = chunk_size - overlap
    windows = [ids[start : start + chunk_size] for start in range(0, len(ids), step)]
    # Drop a trailing window fully contained in the previous one.
    if len(windows) > 1 and len(windows[-1]) <= overlap:
        windows.pop()
    if len(windows) > max_chunks:
        windows = windows[: max_chunks - 1] + [windows[-1]]
    return windows


class EssayCollator:
    """Callable collator used with ``torch.utils.data.DataLoader``."""

    def __init__(self, tokenizer, config: CollatorConfig | None = None, **kwargs):
        if config is None:
            config = CollatorConfig(**kwargs)
        config.validate()
        self.tokenizer = tokenizer
        self.cfg = config
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer has no pad token; set tokenizer.pad_token")
        if self.cls_token_id is None or self.sep_token_id is None:
            raise ValueError("tokenizer must define cls_token and sep_token")

    def _add_special_tokens(self, ids: Sequence[int]) -> List[int]:
        """Build ``[CLS] ids [SEP]`` without relying on a specific tokenizer class.

        Some slow tokenizers (notably ``DebertaV2Tokenizer`` shipped with
        ``microsoft/deberta-v3-*``) do not implement
        ``build_inputs_with_special_tokens``. The single-sequence layout is the
        same for all supported encoders, so falling back to the tokenizer's
        ``cls_token_id``/``sep_token_id`` is exact, not an approximation.
        """
        ids = list(ids)
        builder = getattr(self.tokenizer, "build_inputs_with_special_tokens", None)
        if callable(builder):
            try:
                return list(builder(ids))
            except (AttributeError, NotImplementedError):  # pragma: no cover
                pass

        output: List[int] = []
        if self.cls_token_id is not None:
            output.append(self.cls_token_id)
        output.extend(ids)
        if self.sep_token_id is not None:
            output.append(self.sep_token_id)
        return output

    # ------------------------------------------------------------------ #
    # Per-sample helpers
    # ------------------------------------------------------------------ #
    def _context_ids(self, sample: dict) -> List[int]:
        text = build_context_text(sample["question"], sample["rubric"])
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.cfg.max_context_length:
            ids = truncate_token_ids(
                ids,
                self.cfg.max_context_length,
                strategy="head_tail",
                head_ratio=0.5,
                sep_token_id=self.sep_token_id,
            )
        return ids

    def _answer_chunks(self, sample: dict) -> List[List[int]]:
        ids = self.tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]
        if self.cfg.chunk and len(ids) > self.cfg.max_answer_length:
            windows = make_windows(
                ids,
                chunk_size=self.cfg.chunk_size,
                overlap=self.cfg.chunk_overlap,
                max_chunks=self.cfg.max_chunks,
            )
            # Each window is already <= chunk_size <= max_answer_length.
            windows = [w[: self.cfg.max_answer_length] for w in windows]
            return windows
        ids = truncate_token_ids(
            ids,
            self.cfg.max_answer_length,
            strategy=self.cfg.truncation,
            head_ratio=self.cfg.head_ratio,
            sep_token_id=self.sep_token_id if self.cfg.truncation == "head_tail" else None,
        )
        return [ids]

    # ------------------------------------------------------------------ #
    # Collation
    # ------------------------------------------------------------------ #
    def __call__(self, samples: List[dict]) -> Dict[str, object]:
        if not samples:
            raise ValueError("cannot collate an empty batch")

        labels = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long)
        scores = torch.tensor([int(s["score"]) for s in samples], dtype=torch.long)
        essay_ids = [s["essay_id"] for s in samples]
        prompt_names = [s["prompt_name"] for s in samples]

        # Cross-attention mode: separate answer and context sequences.
        context_ids_batch = [self._context_ids(s) for s in samples]
        answer_chunks = [self._answer_chunks(s) for s in samples]
        max_chunks = max(len(c) for c in answer_chunks)

        context_ids, context_mask = self._pad_sequences(
            context_ids_batch, pad_value=self.pad_token_id, add_special_tokens=True
        )

        padded_answers: List[List[List[int]]] = []
        for chunks in answer_chunks:
            with_specials = [self._add_special_tokens(chunk) for chunk in chunks]
            padded_answers.append(with_specials[:max_chunks])

        answer_ids, answer_mask, chunk_mask = self._pad_nested(
            padded_answers, pad_value=self.pad_token_id
        )
        return {
            "answer_input_ids": answer_ids,
            "answer_attention_mask": answer_mask,
            "context_input_ids": context_ids,
            "context_attention_mask": context_mask,
            "chunk_mask": chunk_mask,
            "labels": labels,
            "scores": scores,
            "essay_ids": essay_ids,
            "prompt_names": prompt_names,
            "n_chunks": torch.tensor([len(c) for c in answer_chunks], dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    # Padding
    # ------------------------------------------------------------------ #
    def _pad_sequences(
        self,
        sequences: List[List[int]],
        pad_value: int,
        add_special_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = []
        for seq in sequences:
            if add_special_tokens:
                # Also handles the empty-context case: CLS/SEP are still added
                # so every sequence has at least one valid token.
                seq = self._add_special_tokens(seq)
            prepared.append(seq)
        max_len = max(max(len(s) for s in prepared), 1)
        ids = torch.full((len(prepared), max_len), pad_value, dtype=torch.long)
        mask = torch.zeros((len(prepared), max_len), dtype=torch.long)
        for i, seq in enumerate(prepared):
            if not seq:
                continue
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
        return ids, mask

    def _pad_nested(
        self,
        sequences: List[List[List[int]]],
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad ``(B, C, L)`` where ``C`` may differ across samples."""
        batch = len(sequences)
        max_chunks = max(len(s) for s in sequences)
        max_len = max(max((len(seq) for sample in sequences for seq in sample), default=1), 1)

        ids = torch.full((batch, max_chunks, max_len), pad_value, dtype=torch.long)
        mask = torch.zeros((batch, max_chunks, max_len), dtype=torch.long)
        chunk_mask = torch.zeros((batch, max_chunks), dtype=torch.long)

        for i, sample in enumerate(sequences):
            for j, seq in enumerate(sample):
                if not seq:
                    continue
                length = min(len(seq), max_len)
                ids[i, j, :length] = torch.tensor(seq[:length], dtype=torch.long)
                mask[i, j, :length] = 1
                chunk_mask[i, j] = 1
        return ids, mask, chunk_mask
