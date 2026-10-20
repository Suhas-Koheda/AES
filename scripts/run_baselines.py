#!/usr/bin/env python
"""Train and evaluate the reference baselines (Models A and B).

Order:
  1. TF-IDF + logistic regression (no GPU needed)
  2. Pretrained Transformer with [QUESTION][RUBRIC][ANSWER] concatenation
  3. (optional) the proposed cross-attention model

Usage
-----
python scripts/run_baselines.py                    # TF-IDF + Transformer
python scripts/run_baselines.py --with-proposed    # also Model C
python scripts/run_baselines.py --only tfidf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "tfidf": "configs/baseline.yaml",
    "transformer": "configs/transformer.yaml",
    "proposed": "configs/cross_attention.yaml",
}


def run_config(path: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_from_config.py"), "--config", path, *extra]
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(CONFIGS), default=None)
    parser.add_argument("--with-proposed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    selected = args.only or ["tfidf", "transformer"]
    if args.with_proposed and "proposed" not in selected:
        selected.append("proposed")

    extra: list[str] = []
    if args.seeds:
        extra += ["--seeds", *[str(s) for s in args.seeds]]
    if args.override:
        extra += ["--override", *args.override]

    for name in selected:
        run_config(CONFIGS[name], extra)

    print("\nAll requested baselines finished. Build tables with:")
    print("  python scripts/make_report.py")


if __name__ == "__main__":
    main()
