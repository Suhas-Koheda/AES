"""The single rubric-conditioned essay cross-attention AES architecture."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .labels import NUM_CLASSES
from .metrics import to_plain_floats
from .utils import count_parameters, get_logger

logger = get_logger()



# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the sequence dimension, ignoring padded positions.

    ``hidden``: (N, L, H); ``mask``: (N, L) with 1 for valid tokens.
    """
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be (N, L, H), got {tuple(hidden.shape)}")
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)  # (N, L, 1)
    summed = (hidden * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts


class MaskedAttentionPooling(nn.Module):
    """Additive attention pooling with padding masks."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.proj(hidden).squeeze(-1)  # (N, L)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        # Guard against all-padding rows (should not happen with special tokens).
        weights = weights * mask.to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights = weights / denom
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)


def build_pooler(pooling: str, hidden_size: int) -> nn.Module:
    if pooling != "attention":
        raise ValueError("only attention pooling is supported")
    return MaskedAttentionPooling(hidden_size)


class ScalarHead(nn.Module):
    """Six-class holistic score head."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(pooled))


class TransformerEncoder(nn.Module):
    """Thin wrapper around a Hugging Face encoder."""

    def __init__(
        self,
        model_name: str,
        freeze: bool = False,
        gradient_checkpointing: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        from transformers import AutoModel

        # The released DeBERTa-v3 checkpoint stores FP16 weights. Loading them
        # as-is leaves FP16 parameters, which AMP's GradScaler cannot unscale.
        # Keep master weights in FP32; autocast handles the FP16 compute.
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                dtype=torch.float32,
            )
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            self.encoder = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32,
            )
        self.encoder.float()
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.model_name = model_name

        if gradient_checkpointing:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                # Cached attention outputs are incompatible with checkpointing.
                if getattr(self.encoder, "config", None) is not None:
                    self.encoder.config.use_cache = False
                try:
                    # Non-reentrant checkpointing does not require the encoder
                    # inputs to carry `requires_grad`.
                    self.encoder.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:  # transformers < 4.36
                    self.encoder.gradient_checkpointing_enable()
            else:  # pragma: no cover
                logger.warning("Encoder does not support gradient checkpointing")

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.freeze = freeze

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


# --------------------------------------------------------------------------- #
# Retained rubric-conditioned cross-attention model
# --------------------------------------------------------------------------- #
class RubricCrossAttentionModel(nn.Module):
    """Answer representations conditioned on question+rubric via cross-attention.

    ``Ha = Encoder(answer_chunk)``
    ``Hc = Encoder([QUESTION] ... [RUBRIC] ...)``
    ``C  = CrossAttention(Q=Ha, K=Hc, V=Hc)``
    ``h  = Ha + sigmoid(W[Ha; C]) * C``
    ``p  = MaskedPool(h)``
    ``logits = Head(p)``

    The context key-padding mask makes padded question/rubric tokens
    ineligible as evidence.  Answer padding is excluded at pooling time.
    """

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        pooling: str = "attention",
        num_attention_heads: int = 8,
        cross_attention_layers: int = 1,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.answer_encoder = TransformerEncoder(
            model_name,
            freeze=freeze_encoder,
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
        )
        self.context_encoder = self.answer_encoder

        hidden = self.answer_encoder.hidden_size
        if hidden % num_attention_heads != 0:
            raise ValueError(
                f"hidden size {hidden} not divisible by {num_attention_heads} heads"
            )

        self.cross_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(cross_attention_layers)
            ]
        )
        self.cross_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(cross_attention_layers)]
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.Sigmoid(),
        )

        self.pooler = build_pooler(pooling, hidden)
        self.head = ScalarHead(hidden, dropout)
        self.pooling = pooling

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        answer_ids = batch["answer_input_ids"]
        answer_mask = batch["answer_attention_mask"]
        context_ids = batch["context_input_ids"]
        context_mask = batch["context_attention_mask"]
        chunk_mask = batch.get("chunk_mask")

        if answer_ids.dim() == 2:
            answer_ids = answer_ids.unsqueeze(1)
            answer_mask = answer_mask.unsqueeze(1)
        if chunk_mask is None:
            chunk_mask = torch.ones(
                answer_ids.shape[:2], dtype=torch.long, device=answer_ids.device
            )

        batch_size, num_chunks, ans_len = answer_ids.shape
        flat_ids = answer_ids.reshape(batch_size * num_chunks, ans_len)
        flat_mask = answer_mask.reshape(batch_size * num_chunks, ans_len)
        valid = chunk_mask.reshape(-1).bool()
        valid_indices = torch.arange(batch_size * num_chunks, device=answer_ids.device)[valid]

        # Encode answer chunks (only valid ones).
        ha = self.answer_encoder(flat_ids[valid], flat_mask[valid])  # (Nv, La, H)
        ha_mask = flat_mask[valid]

        # Encode the shared question+rubric context. The context is identical
        # for every essay of a prompt, so encode each unique prompt context
        # once and broadcast the result across its rows.
        prompt_names = batch.get("prompt_names")
        if prompt_names is not None and len(prompt_names) == batch_size:
            first_row: Dict[str, int] = {}
            for row, name in enumerate(prompt_names):
                first_row.setdefault(str(name), row)
            unique_rows = list(first_row.values())
            if len(unique_rows) < batch_size:
                unique_hc = self.context_encoder(
                    context_ids[unique_rows], context_mask[unique_rows]
                )
                positions = {name: pos for pos, name in enumerate(first_row)}
                remap = torch.tensor(
                    [positions[str(name)] for name in prompt_names],
                    dtype=torch.long,
                    device=answer_ids.device,
                )
                hc = unique_hc[remap]
            else:
                hc = self.context_encoder(context_ids, context_mask)  # (B, Lc, H)
        else:
            hc = self.context_encoder(context_ids, context_mask)  # (B, Lc, H)
        # Expand context to match valid chunk rows.
        batch_index = torch.arange(batch_size, device=answer_ids.device).repeat_interleave(
            num_chunks
        )[valid]
        hc_expanded = hc[batch_index]
        context_mask_expanded = context_mask[batch_index]
        key_padding_mask = context_mask_expanded == 0  # True => ignore

        conditioned = ha
        for attn, norm in zip(self.cross_attention_layers, self.cross_norms):
            attn_out, _ = attn(
                query=conditioned,
                key=hc_expanded,
                value=hc_expanded,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            conditioned = norm(conditioned + attn_out)

        gate = self.gate(torch.cat([ha, conditioned], dim=-1))
        combined = ha + gate * conditioned

        pooled = self.pooler(combined, ha_mask)  # (Nv, H)

        hidden_size = pooled.size(-1)
        out = pooled.new_zeros(batch_size * num_chunks, hidden_size)
        out = out.index_copy(0, valid_indices, pooled)
        out = out.view(batch_size, num_chunks, hidden_size)
        pooled_chunks = masked_mean(out, chunk_mask)

        logits = self.head(pooled_chunks)
        return {
            "logits": logits,
            "pooled": pooled_chunks,
            "cross_attention": None,  # available via need_weights=True if needed
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(model_cfg: dict, cache_dir: Optional[str] = None) -> nn.Module:
    """Build the retained question/rubric-conditioned cross-attention model."""
    if model_cfg.get("name", "cross_attention") != "cross_attention":
        raise ValueError("only model.name='cross_attention' is supported")
    return RubricCrossAttentionModel(
        model_name=model_cfg.get("pretrained_model", "microsoft/deberta-v3-small"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=model_cfg.get("pooling", "attention"),
        num_attention_heads=int(model_cfg.get("num_attention_heads", 8)),
        cross_attention_layers=int(model_cfg.get("cross_attention_layers", 1)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        cache_dir=cache_dir,
    )


def describe_model(model: nn.Module) -> dict:
    return to_plain_floats({
        "class": model.__class__.__name__,
        "parameters": count_parameters(model),
        "encoder": model.answer_encoder.model_name,
        "pooling": model.pooling,
        "fusion": "essay-queries-question-rubric-keys-values; gated residual",
        "score_head": "six-class holistic softmax",
    })
