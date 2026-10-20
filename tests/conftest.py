"""Shared pytest fixtures.

The model-shape tests use a *randomly initialised* tiny DeBERTa and a
dependency-free dummy tokenizer, so they run offline and in seconds. No
pretrained checkpoint or real corpus is required.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class DummyTokenizer:
    """Whitespace tokenizer with deterministic hash ids (offline, picklable)."""

    def __init__(self, vocab_size: int = 256, model_max_length: int = 512):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 2
        self.model_max_length = model_max_length

    def _token_id(self, token: str) -> int:
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        return 3 + digest % (self.vocab_size - 3)

    def __call__(self, text, add_special_tokens: bool = False, **kwargs):
        if isinstance(text, str):
            return {"input_ids": [self._token_id(t) for t in text.split()]}
        return {"input_ids": [[self._token_id(t) for t in doc.split()] for doc in text]}

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_token_id] + list(ids) + [self.sep_token_id]


PROMPTS = ["Prompt A", "Prompt B", "Prompt C"]


def make_synthetic_df(n: int = 120, seed: int = 0, split: str = "train") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        n_words = int(rng.integers(15, 120))
        rows.append(
            {
                "essay_id": f"{split}_{i:05d}",
                "score": int(rng.integers(1, 7)),
                "full_text": " ".join(f"w{int(rng.integers(0, 50))}" for _ in range(n_words)),
                "set": split,
                "pubpriv": 0 if split == "train" else "private",
                "assignment": f"Write an essay for {PROMPTS[i % len(PROMPTS)]}.",
                "prompt_name": PROMPTS[i % len(PROMPTS)],
                "economically_disadvantaged": "Not economically disadvantaged",
                "student_disability_status": "Not identified as having disability",
                "ell_status": "No",
                "race_ethnicity": "White",
                "gender": "F",
                "grade_level": 10,
                "essay_word_count": n_words,
                "task": "Text dependent",
            }
        )
    return pd.DataFrame(rows)


RUBRIC_TEXT = """Holistic Rating Form

SCORE OF 6: Clear and consistent mastery; insightful point of view.

SCORE OF 5: Reasonably consistent mastery; effective point of view.

SCORE OF 4: Adequate mastery; competent point of view.

SCORE OF 3: Developing mastery; inconsistent point of view.

SCORE OF 2: Little mastery; vague point of view.

SCORE OF 1: Very little or no mastery; no viable point of view.
"""


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()


@pytest.fixture
def synthetic_train_df():
    return make_synthetic_df(120, seed=1, split="train")


@pytest.fixture
def synthetic_test_df():
    return make_synthetic_df(60, seed=2, split="test")


@pytest.fixture
def rubric_file(tmp_path):
    path = tmp_path / "rubric.txt"
    path.write_text(RUBRIC_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def patch_tiny_encoder(monkeypatch):
    """Force ``AutoModel.from_pretrained`` to return a tiny random DeBERTa."""
    import transformers
    from transformers import DebertaV2Config, DebertaV2Model

    def _build(*args, **kwargs):
        config = DebertaV2Config(
            vocab_size=256,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
            type_vocab_size=2,
        )
        return DebertaV2Model(config)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(_build))
    return "tiny-random-deberta"


@pytest.fixture
def sample_records():
    return [
        {
            "essay_id": f"e{i}",
            "question": "Write an argument about testing.",
            "answer": " ".join(["evidence"] * 50),
            "rubric": RUBRIC_TEXT,
            "source": None,
            "label": i % 6,
            "score": (i % 6) + 1,
            "prompt_name": PROMPTS[i % len(PROMPTS)],
        }
        for i in range(8)
    ]
