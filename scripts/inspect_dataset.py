#!/usr/bin/env python
"""Inspect the official ASAP 2.0 files and the rubric.

Example
-------
python scripts/inspect_dataset.py --tokenizer microsoft/deberta-v3-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inspect import inspect_dataset  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default="data/asap2_raw/extracted/train/ASAP_2_Final_github_train.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="data/asap2_raw/extracted/test/ASAP_2_Final_github_test.csv",
    )
    parser.add_argument(
        "--rubric",
        default="data/asap2_raw/extracted/rubric/asap_scoring_rubric.txt",
    )
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer for length statistics")
    parser.add_argument(
        "--sample-for-tokenizer",
        type=int,
        default=3000,
        help="number of training essays to tokenise (0 = all)",
    )
    parser.add_argument(
        "--out",
        default="results/inspection/dataset_report.json",
    )
    args = parser.parse_args()

    setup_logging()
    inspect_dataset(
        ROOT / args.train_csv,
        ROOT / args.test_csv,
        ROOT / args.rubric,
        tokenizer_name=args.tokenizer,
        sample_for_tokenizer=args.sample_for_tokenizer or None,
        out_path=ROOT / args.out,
    )


if __name__ == "__main__":
    main()
