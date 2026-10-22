#!/usr/bin/env python
"""Run existing cross-attention model on all ASAP 1.0 prompt-wise folds."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runner import run_experiment  # noqa: E402
from src.utils import load_config, save_json, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asap1_cross_attention.yaml")
    parser.add_argument("--prompt-ids", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging()
    cfg0 = load_config(ROOT / args.config)
    base_out = Path(cfg0["experiment"]["output_dir"])
    if not base_out.is_absolute():
        base_out = ROOT / base_out
    rows = []
    for prompt_id in args.prompt_ids:
        for fold in args.folds:
            cfg = json.loads(json.dumps(cfg0))
            cfg["experiment"].update(seed=args.seed, prompt_id=prompt_id, fold=fold)
            cfg["experiment"]["output_dir"] = str(base_out / f"prompt_{prompt_id}" / f"fold_{fold}")
            result = run_experiment(cfg, root=ROOT)
            metrics = result["test"]["overall"]
            rows.append({"prompt_id": prompt_id, "fold": fold, "seed": args.seed, **metrics})
    if rows:
        summary_path = base_out / "asap1_folds.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        prompt_summary = {}
        for prompt_id in args.prompt_ids:
            prompt_rows = [row for row in rows if row["prompt_id"] == prompt_id]
            prompt_summary[str(prompt_id)] = {}
            for key in ("qwk", "mae", "rmse", "accuracy"):
                vals = [float(row[key]) for row in prompt_rows]
                mean = sum(vals) / len(vals)
                prompt_summary[str(prompt_id)][key] = {
                    "mean": mean,
                    "std": (sum((value - mean) ** 2 for value in vals) / len(vals)) ** 0.5,
                }
        qwk_means = [value["qwk"]["mean"] for key, value in prompt_summary.items()]
        prompt_summary["macro_average"] = {
            "qwk_mean": sum(qwk_means) / len(qwk_means),
            "prompt_count": len(qwk_means),
        }
        save_json(prompt_summary, base_out / "asap1_summary.json")
        print(json.dumps(prompt_summary, indent=2))


if __name__ == "__main__":
    main()
