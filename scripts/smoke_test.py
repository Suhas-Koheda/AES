#!/usr/bin/env python
"""End-to-end smoke test on *synthetic* essays (no official data required).

Creates a tiny fake corpus with the real schema and rubric format, then runs
the full runner (split -> tokenise -> model -> train -> select -> evaluate) with
a randomly initialised tiny DeBERTa. Accuracy is meaningless; the goal is to
prove the pipeline works before downloading a 140M-parameter checkpoint.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runner import run_experiment  # noqa: E402
from src.utils import load_config, merge_overrides, setup_logging  # noqa: E402

PROMPTS = ["Synthetic prompt A", "Synthetic prompt B", "Synthetic prompt C"]

WORDS = (
    "the of and to in is that for it as was with be by on not he this are or "
    "his from at which but have an they you were student essay source evidence "
    "argument claim reason example analysis however therefore conclude"
).split()

RUBRIC = """Holistic Rating Form

SCORE OF 6: An essay in this category demonstrates clear and consistent mastery and effectively develops a point of view with outstanding critical thinking.

SCORE OF 5: An essay in this category demonstrates reasonably consistent mastery and effectively develops a point of view with strong critical thinking.

SCORE OF 4: An essay in this category demonstrates adequate mastery and develops a point of view with competent critical thinking.

SCORE OF 3: An essay in this category demonstrates developing mastery and may develop a point of view inconsistently.

SCORE OF 2: An essay in this category demonstrates little mastery and is flawed by weak critical thinking.

SCORE OF 1: An essay in this category demonstrates very little or no mastery and develops no viable point of view.
"""


def make_row(rng: random.Random, i: int, split: str) -> dict:
    prompt = rng.choice(PROMPTS)
    score = rng.choice([1, 2, 3, 4, 5, 6])
    n_words = rng.randint(40, 400)
    text = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return {
        "essay_id": f"{split}_{i:05d}",
        "score": score,
        "full_text": text,
        "set": split,
        "pubpriv": 0 if split == "train" else rng.choice(["public", "private"]),
        "assignment": f"Write an argument about {prompt}.",
        "prompt_name": prompt,
        "economically_disadvantaged": rng.choice(
            ["Economically disadvantaged", "Not economically disadvantaged"]
        ),
        "student_disability_status": rng.choice(
            ["Identified as having disability", "Not identified as having disability"]
        ),
        "ell_status": rng.choice(["Yes", "No"]),
        "race_ethnicity": rng.choice(["White", "Hispanic/Latino", "Black/African American"]),
        "gender": rng.choice(["F", "M"]),
        "grade_level": rng.choice([6, 8, 9, 10]),
        "essay_word_count": n_words,
        "task": "Text dependent",
    }


def generate(tmp: Path, n_train: int, n_test: int, seed: int = 0) -> tuple[Path, Path, Path]:
    rng = random.Random(seed)
    train = pd.DataFrame([make_row(rng, i, "train") for i in range(n_train)])
    test = pd.DataFrame([make_row(rng, i, "test") for i in range(n_test)])
    train_path = tmp / "train.csv"
    test_path = tmp / "test.csv"
    rubric_path = tmp / "rubric.txt"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    rubric_path.write_text(RUBRIC, encoding="utf-8")
    return train_path, test_path, rubric_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=180)
    parser.add_argument("--n-test", type=int, default=60)
    parser.add_argument("--keep", action="store_true", help="keep the synthetic data directory")
    args = parser.parse_args()

    setup_logging()
    tmp = Path(tempfile.mkdtemp(prefix="aes_smoke_"))
    train_path, test_path, rubric_path = generate(tmp, args.n_train, args.n_test)
    print(f"Synthetic data in {tmp}")

    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg = merge_overrides(
        cfg,
        {
            "experiment": {"output_dir": str(tmp / "results")},
            "data": {
                "train_csv": str(train_path),
                "test_csv": str(test_path),
                "rubric_path": str(rubric_path),
                "max_train_samples": None,
                "max_eval_samples": None,
            },
        },
    )

    results = run_experiment(cfg, root=ROOT)
    print("\nSmoke test completed.")
    print(json.dumps(results["test"]["overall"], indent=2))
    if not args.keep:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
