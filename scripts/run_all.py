#!/usr/bin/env python
"""Run the whole study end to end, in dependency order.

This is the single command to run in Colab::

    python scripts/run_all.py

It runs, in order:

1. dataset inspection (including token-length/truncation statistics)
2. TF-IDF baseline (Model A)
3. Transformer concat baseline (Model B)
4. rubric cross-attention (Model C)
5. cross-attention with the matched 512-token budget (fusion control)
6. CORAL ordinal model
7. CE + expected-score ordinal model
8. the five ablation configurations
9. the held-out-prompt protocol (separate table)
10. the report/tables

Use ``--stages`` to run a subset, ``--seeds`` for multi-seed runs, and
``--override key=value`` to change hyperparameters for every training stage.
Every stage is a separate process, so a failure stops the run with a clear
stage header; completed stages can be skipped by passing ``--stages`` again.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STAGES = {
    "prepare": [PY, "scripts/prepare_data.py", "--download"],
    "inspect": [PY, "scripts/inspect_dataset.py", "--tokenizer", "microsoft/deberta-v3-small"],
    "tfidf": [PY, "scripts/run_from_config.py", "--config", "configs/baseline.yaml"],
    "transformer": [PY, "scripts/run_from_config.py", "--config", "configs/transformer.yaml"],
    "proposed": [PY, "scripts/run_from_config.py", "--config", "configs/cross_attention.yaml"],
    "matched": [PY, "scripts/run_from_config.py", "--config", "configs/cross_attention_matched.yaml"],
    "ordinal": [PY, "scripts/run_from_config.py", "--config", "configs/ordinal_coral.yaml"],
    "ordinal_expected": [PY, "scripts/run_from_config.py", "--config", "configs/ordinal_expected.yaml"],
    "baselines": [PY, "scripts/run_baselines.py", "--only", "tfidf", "transformer"],
    "ablations": [PY, "scripts/run_ablation.py"],
    "cross_prompt": [PY, "scripts/run_cross_prompt.py", "--config", "configs/cross_attention.yaml"],
    "report": [PY, "scripts/make_report.py", "--ablation"],
}

DEFAULT_ORDER = [
    "inspect",
    "tfidf",
    "transformer",
    "proposed",
    "matched",
    "ordinal",
    "ordinal_expected",
    "ablations",
    "cross_prompt",
    "report",
]

# Scripts that accept ``--override`` / ``--seeds``.
TRAINING_SCRIPTS = ("run_from_config.py", "run_baselines.py", "run_ablation.py")
SEED_SCRIPTS = ("run_from_config.py", "run_baselines.py", "run_ablation.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prepare", action="store_true", help="download/extract the official corpus first")
    parser.add_argument("--stages", nargs="+", choices=sorted(STAGES), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    stages = args.stages or list(DEFAULT_ORDER)
    if args.prepare and "prepare" not in stages:
        stages.insert(0, "prepare")

    failed: list[str] = []
    for index, stage in enumerate(stages, start=1):
        cmd = list(STAGES[stage])
        script = Path(cmd[1]).name if len(cmd) > 1 else ""

        if args.seeds and script in SEED_SCRIPTS:
            cmd += ["--seeds", *[str(s) for s in args.seeds]]
        if args.override and script in TRAINING_SCRIPTS:
            cmd += ["--override", *args.override]

        print(
            f"\n{'#' * 70}\n"
            f"# STAGE {index}/{len(stages)}: {stage}\n"
            f"# {' '.join(cmd)}\n"
            f"{'#' * 70}",
            flush=True,
        )
        try:
            env = os.environ.copy()
            # Reduce allocator fragmentation when long sequences and two
            # encoder passes leave large blocks of varying sizes.
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            subprocess.run(cmd, check=True, cwd=ROOT, env=env)
        except subprocess.CalledProcessError as exc:
            failed.append(stage)
            print(
                f"\nStage {stage!r} failed with exit code {exc.returncode}. "
                "Fix the error above, then resume with:\n"
                f"  python scripts/run_all.py --stages {' '.join(stages[index - 1:])}",
                flush=True,
            )
            raise SystemExit(exc.returncode) from exc

    print(
        f"\nAll {len(stages)} stage(s) finished. Results are under {ROOT / 'results'}.\n"
        "Build/refresh the tables with:\n"
        "  python scripts/make_report.py --ablation",
        flush=True,
    )


if __name__ == "__main__":
    main()
