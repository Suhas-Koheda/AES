"""Collator tests: shapes, budgets, truncation and chunking."""

from __future__ import annotations

import torch

from src.collator import CollatorConfig, EssayCollator, make_windows


def _cfg(**kwargs):
    base = dict(
        input_mode="qra_cross",
        max_answer_length=16,
        max_context_length=16,
        max_total_length=40,
    )
    base.update(kwargs)
    cfg = CollatorConfig(**base)
    cfg.validate()
    return cfg


def test_cross_batch_shapes(dummy_tokenizer, sample_records):
    collator = EssayCollator(dummy_tokenizer, _cfg(input_mode="qra_cross"))
    batch = collator(sample_records[:4])
    b, c, la = batch["answer_input_ids"].shape
    assert b == 4 and c == 1
    assert batch["answer_attention_mask"].shape == (b, c, la)
    assert batch["context_input_ids"].shape[0] == 4
    assert batch["labels"].tolist() == [r["label"] for r in sample_records[:4]]
    assert batch["chunk_mask"].tolist() == [[1]] * 4


def test_concat_batch_shapes(dummy_tokenizer, sample_records):
    collator = EssayCollator(
        dummy_tokenizer, _cfg(input_mode="qra_concat", max_context_length=8, max_answer_length=20, max_total_length=40)
    )
    batch = collator(sample_records[:3])
    assert batch["input_ids"].dim() == 3
    assert batch["input_ids"].shape[0] == 3
    assert batch["input_ids"].shape[2] <= 40
    assert torch.all(batch["input_ids"][batch["attention_mask"] == 0] == dummy_tokenizer.pad_token_id)


def test_essay_only_has_no_context(dummy_tokenizer, sample_records):
    collator = EssayCollator(
        dummy_tokenizer, _cfg(input_mode="essay_only", max_context_length=0, max_answer_length=20)
    )
    batch = collator(sample_records[:2])
    assert batch["input_ids"].shape[0] == 2


def test_head_tail_truncation_preserves_last_token(dummy_tokenizer, sample_records):
    collator = EssayCollator(
        dummy_tokenizer,
        _cfg(input_mode="qra_cross", max_answer_length=10, truncation="head_tail"),
    )
    record = dict(sample_records[0])
    record["answer"] = " ".join(f"w{i}" for i in range(100))
    chunk = collator._answer_chunks(record)[0]
    last_content_id = dummy_tokenizer("w99")["input_ids"][0]
    assert last_content_id in chunk
    assert len(chunk) <= 10


def test_head_truncation_keeps_first_token(dummy_tokenizer, sample_records):
    collator = EssayCollator(
        dummy_tokenizer,
        _cfg(input_mode="qra_cross", max_answer_length=10, truncation="head"),
    )
    record = dict(sample_records[0])
    record["answer"] = " ".join(f"w{i}" for i in range(100))
    chunk = collator._answer_chunks(record)[0]
    first_id = dummy_tokenizer("w0")["input_ids"][0]
    assert chunk[0] == first_id
    assert len(chunk) == 10


def test_context_is_never_empty_for_cross(dummy_tokenizer, sample_records):
    collator = EssayCollator(dummy_tokenizer, _cfg(input_mode="qra_cross"))
    batch = collator(sample_records[:1])
    assert int(batch["context_attention_mask"].sum()) >= 2  # CLS + SEP at minimum


def test_chunking_caps_and_keeps_last_window(dummy_tokenizer, sample_records):
    collator = EssayCollator(
        dummy_tokenizer,
        _cfg(
            input_mode="qra_cross",
            max_answer_length=24,
            chunk=True,
            chunk_size=24,
            chunk_overlap=4,
            max_chunks=3,
        ),
    )
    record = dict(sample_records[0])
    record["answer"] = " ".join(f"w{i}" for i in range(400))
    chunks = collator._answer_chunks(record)
    assert len(chunks) == 3
    last_id = dummy_tokenizer("w399")["input_ids"][0]
    assert last_id in chunks[-1]


def test_make_windows_short_input():
    assert make_windows(list(range(5)), chunk_size=10, overlap=2, max_chunks=4) == [list(range(5))]


def test_make_windows_overlap():
    windows = make_windows(list(range(30)), chunk_size=10, overlap=5, max_chunks=10)
    assert len(windows) > 1
    assert all(len(w) <= 10 for w in windows)
    # Consecutive windows overlap by 5.
    assert windows[0][-5:] == windows[1][:5]


def test_invalid_config_rejected():
    try:
        _cfg(input_mode="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_concat_budget_validation():
    try:
        _cfg(input_mode="qra_concat", max_context_length=30, max_answer_length=30, max_total_length=40)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for over-budget config")
