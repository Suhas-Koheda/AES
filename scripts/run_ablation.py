#!/usr/bin/env python
"""Run the controlled ablation matrix on identical splits.

Rows
----
1. essay only                         (configs/essay_only.yaml)
2. question + answer                  (configs/qa_concat.yaml)
3. question + rubric + answer, concat (configs/qra_concat.yaml)
4. question + rubric + answer, cross  (configs/qra_cross.yaml)
5. cross-attention + ordinal objective(configs/qra_cross_coral.yaml)

A sixth, controlled row uses the *same token budgets* as row 3 but swaps
concatenation for cross-attention (configs/cross_attention_matched.yaml), so
the fusion mechanism is compared without also changing how much text is seen.

Usage
-----
python scripts/run_ablation.py
python scripts/run_ablation.py --configs qra_concat qra_cross --seeds 42 43
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ABLATIONS = [
    ("essay_only", "Essay only", "configs/essay_only.yaml"),
    ("qa_concat", "Question + answer", "configs/qa_concat.yaml"),
    ("qra_concat", "Question + rubric + answer (concat)", "configs/qra_concat.yaml"),
    ("qra_cross", "Question + rubric + answer (cross-attention)", "configs/qra_cross.yaml"),
    ("qra_cross_coral", "Cross-attention + ordinal (CORAL)", "configs/qra_cross_coral.yaml"),
    ("cross_attention_matched", "Cross-attention, matched token budget", "configs/cross_attention_matched.yaml"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[key for key, _, _ in ABLATIONS])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected = [row for row in ABLATIONS if row[0] in args.configs]
    if not selected:
        raise SystemExit(f"unknown configs: {args.configs}")

    for key, label, config in selected:
        print(f"\n{'=' * 70}\nAblation: {label}\n{'=' * 70}", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "run_from_config.py"), "--config", config]
        if args.seeds:
            cmd += ["--seeds", *[str(s) for s in args.seeds]]
        if args.override:
            cmd += ["--override", *args.override]
        if args.skip_existing:
            summary_guess = ROOT / "results" / _guess_dir(key) / "summary.json"
            if summary_guess.exists():
                print(f"skipping (found {summary_guess})")
                continue
        subprocess.run(cmd, check=True, cwd=ROOT)

    print("\nAblations finished. Build the table with:")
    print("  python scripts/make_report.py --ablation")


def _guess_dir(key: str) -> str:
    from configs_map import OUTPUT_DIRS  # type: ignore

    return OUTPUT_DIRS.get(key, key)


if __name__ == "__main__":
    main()
