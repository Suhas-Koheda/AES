"""Model shape, masking and gradient tests using a tiny random encoder."""

from __future__ import annotations

import torch

from src.collator import CollatorConfig, EssayCollator
from src.models import (
    RubricCrossAttentionModel,
    TransformerBaseline,
    build_model,
    masked_mean,
)

HIDDEN = 32
NUM_HEADS = 4


def _collator(tokenizer, input_mode: str):
    cfg = CollatorConfig(
        input_mode=input_mode,
        max_answer_length=20,
        max_context_length=16,
        max_total_length=40,
    )
    cfg.validate()
    return EssayCollator(tokenizer, cfg)


def _baseline(patch_tiny_encoder, head_type="softmax", pooling="attention"):
    return TransformerBaseline(
        model_name=patch_tiny_encoder,
        pooling=pooling,
        head_type=head_type,
        dropout=0.0,
    )


def _cross(patch_tiny_encoder, head_type="softmax", combine="gate"):
    return RubricCrossAttentionModel(
        model_name=patch_tiny_encoder,
        pooling="attention",
        head_type=head_type,
        num_attention_heads=NUM_HEADS,
        combine=combine,
        dropout=0.0,
    )


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(masked_mean(hidden, mask), torch.tensor([[2.0]]))


def test_baseline_forward_shape(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_concat")
    batch = collator(sample_records[:4])
    model = _baseline(patch_tiny_encoder)
    out = model(batch)
    assert out["logits"].shape == (4, 6)
    assert out["pooled"].shape == (4, HIDDEN)


def test_cross_forward_shape(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_cross")
    batch = collator(sample_records[:4])
    model = _cross(patch_tiny_encoder)
    out = model(batch)
    assert out["logits"].shape == (4, 6)
    assert out["pooled"].shape == (4, HIDDEN)


def test_coral_head_shape(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_cross")
    batch = collator(sample_records[:3])
    model = _cross(patch_tiny_encoder, head_type="coral")
    out = model(batch)
    assert out["logits"].shape == (3, 5)


def test_all_pooling_types(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_concat")
    batch = collator(sample_records[:2])
    for pooling in ("mean", "cls", "attention"):
        model = _baseline(patch_tiny_encoder, pooling=pooling)
        assert model(batch)["logits"].shape == (2, 6)


def test_all_combine_types(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_cross")
    batch = collator(sample_records[:2])
    for combine in ("residual", "gate", "concat_project"):
        model = _cross(patch_tiny_encoder, combine=combine)
        assert model(batch)["logits"].shape == (2, 6)


def test_context_padding_does_not_change_output(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_cross")
    batch = collator(sample_records[:3])
    model = _cross(patch_tiny_encoder).eval()
    with torch.no_grad():
        base = model(batch)["logits"]

        b, length = batch["context_input_ids"].shape
        pad_ids = torch.full((b, 5), dummy_tokenizer.pad_token_id, dtype=torch.long)
        pad_mask = torch.zeros((b, 5), dtype=torch.long)
        extended = dict(batch)
        extended["context_input_ids"] = torch.cat([batch["context_input_ids"], pad_ids], dim=1)
        extended["context_attention_mask"] = torch.cat(
            [batch["context_attention_mask"], pad_mask], dim=1
        )
        other = model(extended)["logits"]
    assert torch.allclose(base, other, atol=1e-5), (base - other).abs().max()


def test_answer_padding_does_not_change_output(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_cross")
    batch = collator(sample_records[:3])
    model = _cross(patch_tiny_encoder).eval()
    with torch.no_grad():
        base = model(batch)["logits"]

        b, c, length = batch["answer_input_ids"].shape
        pad_ids = torch.full((b, c, 5), dummy_tokenizer.pad_token_id, dtype=torch.long)
        pad_mask = torch.zeros((b, c, 5), dtype=torch.long)
        extended = dict(batch)
        extended["answer_input_ids"] = torch.cat([batch["answer_input_ids"], pad_ids], dim=2)
        extended["answer_attention_mask"] = torch.cat(
            [batch["answer_attention_mask"], pad_mask], dim=2
        )
        other = model(extended)["logits"]
    assert torch.allclose(base, other, atol=1e-5), (base - other).abs().max()


def test_chunked_batch_forward(dummy_tokenizer, sample_records, patch_tiny_encoder):
    cfg = CollatorConfig(
        input_mode="qra_cross",
        max_answer_length=12,
        max_context_length=16,
        chunk=True,
        chunk_size=12,
        chunk_overlap=4,
        max_chunks=3,
    )
    cfg.validate()
    collator = EssayCollator(dummy_tokenizer, cfg)
    records = [dict(r) for r in sample_records[:2]]
    for record in records:
        record["answer"] = " ".join(f"w{i}" for i in range(120))
    batch = collator(records)
    assert batch["answer_input_ids"].shape[1] == 3
    model = _cross(patch_tiny_encoder)
    assert model(batch)["logits"].shape == (2, 6)


def test_gradients_flow(dummy_tokenizer, sample_records, patch_tiny_encoder):
    collator = _collator(dummy_tokenizer, "qra_cross")
    batch = collator(sample_records[:4])
    model = _cross(patch_tiny_encoder)
    out = model(batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], batch["labels"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_build_model_factory(patch_tiny_encoder):
    model = build_model(
        {
            "name": "cross_attention",
            "pretrained_model": patch_tiny_encoder,
            "num_attention_heads": NUM_HEADS,
            "head_type": "softmax",
        }
    )
    assert isinstance(model, RubricCrossAttentionModel)
    model2 = build_model({"name": "transformer", "pretrained_model": patch_tiny_encoder})
    assert isinstance(model2, TransformerBaseline)


def test_build_model_rejects_unknown(patch_tiny_encoder):
    try:
        build_model({"name": "nope", "pretrained_model": patch_tiny_encoder})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_shared_encoder_is_actually_shared(patch_tiny_encoder):
    model = _cross(patch_tiny_encoder)
    assert model.context_encoder is model.answer_encoder
