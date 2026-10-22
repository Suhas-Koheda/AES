"""Cross-attention batch collation, truncation and chunking tests."""

import pytest

from src.collator import CollatorConfig, EssayCollator, make_windows


def collator(tokenizer, **kwargs):
    defaults = {"max_answer_length": 16, "max_context_length": 16}
    defaults.update(kwargs)
    cfg = CollatorConfig(**defaults)
    return EssayCollator(tokenizer, cfg)


def test_cross_attention_batch_shapes(dummy_tokenizer, sample_records):
    batch = collator(dummy_tokenizer)(sample_records[:4])
    b, chunks, _ = batch["answer_input_ids"].shape
    assert (b, chunks) == (4, 1)
    assert batch["context_input_ids"].shape[0] == 4
    assert batch["labels"].tolist() == [r["label"] for r in sample_records[:4]]
    assert batch["chunk_mask"].tolist() == [[1]] * 4


def test_truncation_preserves_essay_ending(dummy_tokenizer, sample_records):
    c = collator(dummy_tokenizer, max_answer_length=10, truncation="head_tail")
    row = dict(sample_records[0], answer=" ".join(f"w{i}" for i in range(100)))
    answer = c._answer_chunks(row)[0]
    assert dummy_tokenizer("w99")["input_ids"][0] in answer
    assert len(answer) <= 10


def test_context_includes_question_and_rubric(dummy_tokenizer, sample_records):
    batch = collator(dummy_tokenizer)(sample_records[:1])
    assert int(batch["context_attention_mask"].sum()) >= 2


def test_chunking_caps_and_keeps_last_window(dummy_tokenizer, sample_records):
    c = collator(dummy_tokenizer, max_answer_length=24, chunk=True,
                 chunk_size=24, chunk_overlap=4, max_chunks=3)
    row = dict(sample_records[0], answer=" ".join(f"w{i}" for i in range(400)))
    chunks = c._answer_chunks(row)
    assert len(chunks) == 3
    assert dummy_tokenizer("w399")["input_ids"][0] in chunks[-1]


def test_window_overlap():
    windows = make_windows(list(range(30)), chunk_size=10, overlap=5, max_chunks=10)
    assert windows[0][-5:] == windows[1][:5]


def test_unsupported_input_mode_rejected():
    with pytest.raises(ValueError):
        CollatorConfig(input_mode="unsupported").validate()


def test_padding_token_builder_fallback(dummy_tokenizer, sample_records):
    class TokenizerWithoutBuilder:
        pad_token_id = dummy_tokenizer.pad_token_id
        cls_token_id = dummy_tokenizer.cls_token_id
        sep_token_id = dummy_tokenizer.sep_token_id

        def __call__(self, *args, **kwargs):
            return dummy_tokenizer(*args, **kwargs)

    batch = collator(TokenizerWithoutBuilder())(sample_records[:1])
    valid = int(batch["context_attention_mask"][0].sum())
    assert batch["context_input_ids"][0, 0].item() == dummy_tokenizer.cls_token_id
    assert batch["context_input_ids"][0, valid - 1].item() == dummy_tokenizer.sep_token_id
