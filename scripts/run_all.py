#!/usr/bin/env python
"""Run the whole study end to end.

Intended for a single GPU session. Each stage is skippable and resumable; pass
``--stages`` to choose a subset. Results are written under ``results/``.

Example
-------
python scripts/run_all.py --prepare --seeds 42 43 44
python scripts/run_all.py --stages baselines proposed ablations
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGES = {
    "prepare": ["python", "scripts/prepare_data.py", "--download"],
    "inspect": ["python", "scripts/inspect_dataset.py", "--tokenizer", "microsoft/deberta-v3-small"],
    "baselines": ["python", "scripts/run_baselines.py"],
    "proposed": ["python", "scripts/run_from_config.py", "--config", "configs/cross_attention.yaml"],
    "ordinal": ["python", "scripts/run_from_config.py", "--config", "configs/ordinal_coral.yaml"],
    "ordinal_expected": ["python", "scripts/run_from_config.py", "--config", "configs/ordinal_expected.yaml"],
    "ablations": ["python", "scripts/run_ablation.py"],
    "cross_prompt": ["python", "scripts/run_cross_prompt.py", "--config", "configs/cross_attention.yaml"],
    "report": ["python", "scripts/make_report.py", "--ablation"],
}

DEFAULT_ORDER = [
    "inspect",
    "baselines",
    "proposed",
    "ordinal",
    "ordinal_expected",
    "ablations",
    "cross_prompt",
    "report",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="download/extract the official corpus first")
    parser.add_argument("--stages", nargs="+", choices=sorted(STAGES), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    stages = args.stages or list(DEFAULT_ORDER)
    if args.prepare and "prepare" not in stages:
        stages.insert(0, "prepare")

    for stage in stages:
        cmd = list(STAGES[stage])
        if args.seeds and cmd[1].endswith(("run_from_config.py", "run_baselines.py")):
            cmd += ["--seeds", *[str(s) for s in args.seeds]]
        if args.override and cmd[1].endswith("run_from_config.py"):
            cmd += ["--override", *args.override]
        print(f"\n{'#' * 70}\n# STAGE: {stage}\n# {' '.join(cmd)}\n{'#' * 70}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
