#!/usr/bin/env python
"""Held-out-prompt generalization sweep (Evaluation 2).

Each prompt is removed from training *and* validation, the model is trained on
the remaining prompts, selected on a validation split of those prompts, and
evaluated on the held-out prompt only.

This protocol uses the official *training* partition as its pool because the
released ASAP 2.0 test file contains only five of the seven prompts. It never
reads official test labels.

Usage
-----
python scripts/run_cross_prompt.py --config configs/cross_attention.yaml
python scripts/run_cross_prompt.py --config configs/cross_attention.yaml \
    --prompts "Driverless cars" "The Face on Mars"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.generalization import run_heldout_prompt_sweep  # noqa: E402
from src.runner import run_experiment  # noqa: E402
from src.utils import load_config, merge_overrides, parse_key_value_overrides, setup_logging  # noqa: E402


def available_prompts(train_csv: Path) -> list[str]:
    df = pd.read_csv(train_csv, usecols=["prompt_name"])
    return sorted(df["prompt_name"].unique().tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cross_attention.yaml")
    parser.add_argument("--prompts", nargs="+", default=None, help="subset of prompts to hold out")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(ROOT / args.config)
    if args.override:
        cfg = merge_overrides(cfg, parse_key_value_overrides(args.override))
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed

    train_csv = ROOT / cfg["data"]["train_csv"]
    prompts = args.prompts or available_prompts(train_csv)
    unknown = [p for p in prompts if p not in available_prompts(train_csv)]
    if unknown:
        raise SystemExit(f"unknown prompts: {unknown}")

    base_name = cfg["experiment"].get("name", "run")
    output_root = args.output_root or f"results/cross_prompt/{base_name}"

    print(f"Holding out {len(prompts)} prompts: {prompts}", flush=True)
    payload = run_heldout_prompt_sweep(
        cfg, prompts, ROOT / output_root, runner=lambda c: run_experiment(c, root=ROOT)
    )
    aggregate = payload.get("aggregate", {})
    print("\n=== Held-out-prompt aggregate ===")
    for metric in ("qwk", "mae", "accuracy", "spearman"):
        stats = aggregate.get(metric, {})
        if stats:
            print(
                f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
                f"(min {stats['min']:.4f}, max {stats['max']:.4f})"
            )


if __name__ == "__main__":
    main()
