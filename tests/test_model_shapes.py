"""Shape, masking, gradients and construction for the retained model."""

import torch

from src.collator import CollatorConfig, EssayCollator
from src.models import RubricCrossAttentionModel, build_model, masked_mean

HIDDEN = 32
NUM_HEADS = 4


def make_collator(tokenizer):
    return EssayCollator(tokenizer, CollatorConfig(
        max_answer_length=20, max_context_length=16,
    ))


def make_model(tiny_encoder):
    return RubricCrossAttentionModel(
        model_name=tiny_encoder,
        pooling="attention",
        num_attention_heads=NUM_HEADS,
        dropout=0.0,
    )


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert torch.allclose(masked_mean(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[2.0]]))


def test_cross_attention_forward_shapes(dummy_tokenizer, sample_records, patch_tiny_encoder):
    batch = make_collator(dummy_tokenizer)(sample_records[:4])
    output = make_model(patch_tiny_encoder)(batch)
    assert output["logits"].shape == (4, 6)
    assert output["pooled"].shape == (4, HIDDEN)


def test_attention_masks_ignore_appended_padding(dummy_tokenizer, sample_records, patch_tiny_encoder):
    batch = make_collator(dummy_tokenizer)(sample_records[:3])
    model = make_model(patch_tiny_encoder).eval()
    with torch.no_grad():
        original_logits = model(batch)["logits"]
        extended = dict(batch)
        for ids_key, mask_key in (("context_input_ids", "context_attention_mask"),
                                  ("answer_input_ids", "answer_attention_mask")):
            ids, mask = batch[ids_key], batch[mask_key]
            pad_shape = list(ids.shape)
            pad_shape[-1] = 5
            extended[ids_key] = torch.cat([ids, torch.zeros(pad_shape, dtype=ids.dtype)], dim=-1)
            extended[mask_key] = torch.cat([mask, torch.zeros(pad_shape, dtype=mask.dtype)], dim=-1)
        assert torch.allclose(original_logits, model(extended)["logits"], atol=1e-5)


def test_loss_has_gradients(dummy_tokenizer, sample_records, patch_tiny_encoder):
    batch = make_collator(dummy_tokenizer)(sample_records[:4])
    model = make_model(patch_tiny_encoder)
    torch.nn.functional.cross_entropy(model(batch)["logits"], batch["labels"]).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def test_model_factory_only_builds_cross_attention(patch_tiny_encoder):
    model = build_model({"name": "cross_attention", "pretrained_model": patch_tiny_encoder,
                         "num_attention_heads": NUM_HEADS})
    assert isinstance(model, RubricCrossAttentionModel)
    try:
        build_model({"name": "transformer", "pretrained_model": patch_tiny_encoder})
    except ValueError:
        pass
    else:
        raise AssertionError("non-cross-attention architecture was accepted")
