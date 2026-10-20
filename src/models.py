"""Models for rubric-conditioned Automated Essay Scoring.

Three families are implemented so that they can be compared on identical
splits:

``TfidfEssayScorer`` (Model A)
    TF-IDF + a conventional classifier.  Cheap reference point.

``TransformerBaseline`` (Model B)
    A pretrained encoder over a single concatenated sequence
    ``[QUESTION] ... [RUBRIC] ... [ANSWER] ...`` with a 6-way head.  The same
    class also covers the essay-only and question+answer ablations.

``RubricCrossAttentionModel`` (Model C)
    A shared pretrained encoder produced separate answer and
    question+rubric representations.  Answer tokens attend to the context via
    multi-head cross-attention; the attended representation is combined with
    the original answer representation using a gate; masked pooling then feeds
    a 6-way (or ordinal) head.

``head_type`` selects a plain softmax head or a CORAL ordinal head.  The
loss/decoding for each is kept mathematically consistent in ``losses.py`` and
``labels.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import NUM_CLASSES
from .metrics import to_plain_floats
from .utils import count_parameters, get_logger

logger = get_logger()

POOLING_TYPES = {"mean", "cls", "attention"}
COMBINE_TYPES = {"residual", "gate", "concat_project"}
HEAD_TYPES = {"softmax", "coral"}


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
    if pooling not in POOLING_TYPES:
        raise ValueError(f"pooling must be one of {sorted(POOLING_TYPES)}, got {pooling!r}")
    if pooling == "attention":
        return MaskedAttentionPooling(hidden_size)
    return _SimplePooler(pooling)


class _SimplePooler(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.kind == "mean":
            return masked_mean(hidden, mask)
        if self.kind == "cls":
            return hidden[:, 0]
        raise ValueError(self.kind)


class ScalarHead(nn.Module):
    """Task head producing either 6 softmax logits or 5 CORAL logits."""

    def __init__(self, hidden_size: int, head_type: str = "softmax", dropout: float = 0.1):
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {sorted(HEAD_TYPES)}")
        self.head_type = head_type
        self.dropout = nn.Dropout(dropout)
        if head_type == "softmax":
            self.classifier = nn.Linear(hidden_size, NUM_CLASSES)
        else:
            # CORAL: one shared weight vector + K-1 ordered biases.
            self.coral_weight = nn.Linear(hidden_size, 1, bias=False)
            self.coral_bias = nn.Parameter(torch.linspace(-1.0, 1.0, NUM_CLASSES - 1))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = self.dropout(pooled)
        if self.head_type == "softmax":
            return self.classifier(pooled)
        # (N, 1) + (K-1,) -> (N, K-1)
        return self.coral_weight(pooled) + self.coral_bias.unsqueeze(0)


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
# Model B: concatenation baseline
# --------------------------------------------------------------------------- #
class TransformerBaseline(nn.Module):
    """Pretrained encoder over a single concatenated sequence.

    Handles optional chunking: input is ``(B, C, L)`` where ``C`` is the number
    of essay chunks (1 when chunking is disabled).  Chunk representations are
    mean-pooled with the chunk mask.
    """

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        pooling: str = "attention",
        head_type: str = "softmax",
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            model_name,
            freeze=freeze_encoder,
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
        )
        self.pooler = build_pooler(pooling, self.encoder.hidden_size)
        self.head = ScalarHead(self.encoder.hidden_size, head_type, dropout)
        self.pooling = pooling
        self.head_type = head_type

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        chunk_mask = batch.get("chunk_mask")

        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
        if chunk_mask is None:
            chunk_mask = torch.ones(
                input_ids.shape[:2], dtype=torch.long, device=input_ids.device
            )

        batch_size, num_chunks, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_chunks, seq_len)
        flat_mask = attention_mask.reshape(batch_size * num_chunks, seq_len)
        valid = chunk_mask.reshape(-1).bool()

        hidden = self.encoder(flat_ids[valid], flat_mask[valid])
        pooled = self.pooler(hidden, flat_mask[valid])  # (Nv, H)

        out = pooled.new_zeros(batch_size * num_chunks, pooled.size(-1))
        out = out.index_copy(0, torch.arange(batch_size * num_chunks, device=pooled.device)[valid], pooled)
        out = out.view(batch_size, num_chunks, pooled.size(-1))
        pooled_chunks = masked_mean(out, chunk_mask)

        logits = self.head(pooled_chunks)
        return {"logits": logits, "pooled": pooled_chunks}


# --------------------------------------------------------------------------- #
# Model C: rubric-conditioned cross-attention
# --------------------------------------------------------------------------- #
class RubricCrossAttentionModel(nn.Module):
    """Answer representations conditioned on question+rubric via cross-attention.

    ``Ha = Encoder(answer_chunk)``
    ``Hc = Encoder([QUESTION] ... [RUBRIC] ...)``
    ``C  = CrossAttention(Q=Ha, K=Hc, V=Hc)``
    ``h  = Combine(Ha, C)``  (residual / gate / concat-projection)
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
        head_type: str = "softmax",
        num_attention_heads: int = 8,
        cross_attention_layers: int = 1,
        combine: str = "gate",
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        share_encoder: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        if combine not in COMBINE_TYPES:
            raise ValueError(f"combine must be one of {sorted(COMBINE_TYPES)}")
        self.combine_type = combine
        self.share_encoder = share_encoder

        self.answer_encoder = TransformerEncoder(
            model_name,
            freeze=freeze_encoder,
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
        )
        if share_encoder:
            self.context_encoder = self.answer_encoder
        else:
            self.context_encoder = TransformerEncoder(
                model_name,
                freeze=freeze_encoder,
                gradient_checkpointing=gradient_checkpointing,
                cache_dir=cache_dir,
            )

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

        if combine == "gate":
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.Sigmoid(),
            )
        elif combine == "concat_project":
            self.combine_proj = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
        else:
            self.combine_norm = nn.LayerNorm(hidden)

        self.pooler = build_pooler(pooling, hidden)
        self.head = ScalarHead(hidden, head_type, dropout)
        self.pooling = pooling
        self.head_type = head_type

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

        # Encode the shared question+rubric context once per essay.
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

        if self.combine_type == "residual":
            combined = self.combine_norm(ha + conditioned)
        elif self.combine_type == "gate":
            gate = self.gate(torch.cat([ha, conditioned], dim=-1))
            combined = ha + gate * conditioned
        else:  # concat_project
            combined = self.combine_proj(torch.cat([ha, conditioned], dim=-1))

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
# Model A: TF-IDF baseline (sklearn)
# --------------------------------------------------------------------------- #
class TfidfEssayScorer:
    """TF-IDF + conventional classifier.

    ``classifier="logreg"`` gives calibrated-ish 6-class probabilities.
    ``classifier="ridge"`` is a regression baseline; its continuous outputs are
    rounded and clipped to classes 0..5 (documented in ``README``).
    ``classifier="svc"`` uses a linear SVM with a softmax over decision
    functions as a pseudo-probability (only used for argmax decoding).
    """

    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 100_000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        classifier: str = "logreg",
        C: float = 1.0,
        class_weight: str | None = "balanced",
        max_iter: int = 3000,
        seed: int = 42,
        lowercase: bool = True,
    ):
        self.analyzer = analyzer
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.classifier_name = classifier
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.seed = seed
        self.lowercase = lowercase
        self.vectorizer = None
        self.classifier = None

    # -- construction -------------------------------------------------- #
    def _build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=self.lowercase,
            dtype=np.float32,
        )

        if self.classifier_name == "logreg":
            from sklearn.linear_model import LogisticRegression

            self.classifier = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                random_state=self.seed,
                n_jobs=-1,
            )
        elif self.classifier_name == "ridge":
            from sklearn.linear_model import Ridge

            self.classifier = Ridge(alpha=1.0 / max(self.C, 1e-6), random_state=self.seed)
        elif self.classifier_name == "svc":
            from sklearn.svm import LinearSVC

            self.classifier = LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.seed,
                max_iter=self.max_iter,
            )
        else:
            raise ValueError(f"unknown classifier: {self.classifier_name!r}")

    # -- sklearn API ---------------------------------------------------- #
    def fit(self, texts, labels):
        labels = np.asarray(labels, dtype=np.int64)
        self._build()
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        return self

    def decision_function(self, texts) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        if self.classifier_name == "ridge":
            raw = np.asarray(self.classifier.predict(X), dtype=np.float64).ravel()
            # Convert the regression output back to a 6-class logit-like space
            # so that downstream decoding is uniform.  Not a probability.
            class_centers = np.arange(NUM_CLASSES, dtype=np.float64)
            return -((raw[:, None] - class_centers[None, :]) ** 2)
        if self.classifier_name == "logreg":
            return np.asarray(self.classifier.predict_proba(X), dtype=np.float64)
        # LinearSVC
        return np.asarray(self.classifier.decision_function(X), dtype=np.float64)

    def predict_proba(self, texts) -> np.ndarray:
        """Return ``(N, 6)`` class probabilities (softmax over decision values)."""
        raw = self.decision_function(texts)
        if self.classifier_name == "logreg":
            probs = raw
        else:
            raw = raw - raw.max(axis=1, keepdims=True)
            exp = np.exp(raw)
            probs = exp / exp.sum(axis=1, keepdims=True)
        # Some classifiers may not produce all six columns on tiny fits.
        if probs.shape[1] != NUM_CLASSES:
            full = np.zeros((probs.shape[0], NUM_CLASSES), dtype=np.float64)
            present = list(self.classifier.classes_)
            full[:, present] = probs
            probs = full
        return probs

    def predict(self, texts) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1).astype(np.int64)

    # -- persistence ---------------------------------------------------- #
    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TfidfEssayScorer":
        import joblib

        return joblib.load(path)

    def parameter_count(self) -> int:
        if self.vectorizer is None or self.classifier is None:
            return 0
        n = len(self.vectorizer.vocabulary_)
        if self.classifier_name == "logreg":
            n += int(np.prod(self.classifier.coef_.shape)) + int(self.classifier.intercept_.size)
        elif hasattr(self.classifier, "coef_"):
            n += int(np.prod(self.classifier.coef_.shape))
        return int(n)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(model_cfg: dict, cache_dir: Optional[str] = None) -> nn.Module:
    """Build a neural model from a config sub-dict."""
    name = model_cfg.get("name", "cross_attention")
    head_type = model_cfg.get("head_type", "softmax")
    dropout = float(model_cfg.get("dropout", 0.1))
    pooling = model_cfg.get("pooling", "attention")
    pretrained = model_cfg.get("pretrained_model", "microsoft/deberta-v3-small")

    if name in {"transformer", "baseline"}:
        return TransformerBaseline(
            model_name=pretrained,
            dropout=dropout,
            pooling=pooling,
            head_type=head_type,
            freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
            cache_dir=cache_dir,
        )
    if name in {"cross_attention", "cross_attn", "rubric_cross_attention"}:
        return RubricCrossAttentionModel(
            model_name=pretrained,
            dropout=dropout,
            pooling=pooling,
            head_type=head_type,
            num_attention_heads=int(model_cfg.get("num_attention_heads", 8)),
            cross_attention_layers=int(model_cfg.get("cross_attention_layers", 1)),
            combine=model_cfg.get("combine", "gate"),
            freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
            share_encoder=bool(model_cfg.get("share_encoder", True)),
            cache_dir=cache_dir,
        )
    raise ValueError(f"unknown model name: {name!r}")


def describe_model(model: nn.Module) -> dict:
    counts = count_parameters(model)
    info = {
        "class": model.__class__.__name__,
        "parameters": counts,
        "head_type": getattr(model, "head_type", None),
    }
    if isinstance(model, (TransformerBaseline, RubricCrossAttentionModel)):
        info["encoder"] = model.encoder.model_name if hasattr(model, "encoder") else model.answer_encoder.model_name
        info["pooling"] = model.pooling
    return to_plain_floats(info)
