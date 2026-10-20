#!/usr/bin/env python
"""Run one configuration (optionally across several seeds).

Examples
--------
python scripts/run_from_config.py --config configs/baseline.yaml
python scripts/run_from_config.py --config configs/cross_attention.yaml --seeds 42 43 44
python scripts/run_from_config.py --config configs/transformer.yaml \
    --override train.epochs=2 data.max_train_samples=2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runner import run_experiment  # noqa: E402
from src.utils import (  # noqa: E402
    load_config,
    merge_overrides,
    parse_key_value_overrides,
    save_json,
    setup_logging,
)


def aggregate_seeds(seed_dirs: list[Path], out_path: Path) -> dict:
    """Aggregate overall test/val metrics across seeds (mean/std)."""
    import numpy as np

    rows = []
    for directory in seed_dirs:
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, encoding="utf-8") as fh:
            rows.append(json.load(fh))

    if not rows:
        return {}

    keys = ["val_qwk", "test_qwk", "test_mae", "test_rmse", "test_accuracy", "test_spearman"]
    aggregate = {"n_seeds": len(rows), "seeds": [r.get("seed") for r in rows], "metrics": {}}
    for key in keys:
        values = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        aggregate["metrics"][key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "n_defined": int(finite.size),
        }
    aggregate["per_seed"] = rows
    save_json(aggregate, out_path)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="run several seeds")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="dotted key=value overrides, e.g. train.epochs=2",
    )
    parser.add_argument("--output-dir", default=None, help="override experiment.output_dir")
    args = parser.parse_args()

    setup_logging()
    base_cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    if args.override:
        base_cfg = merge_overrides(base_cfg, parse_key_value_overrides(args.override))
    if args.output_dir:
        base_cfg.setdefault("experiment", {})["output_dir"] = args.output_dir

    seeds = args.seeds or ([args.seed] if args.seed is not None else [base_cfg["experiment"].get("seed", 42)])
    multi = len(seeds) > 1
    base_out = Path(base_cfg["experiment"].get("output_dir", "results/run"))
    if not base_out.is_absolute():
        base_out = ROOT / base_out

    seed_dirs = []
    for seed in seeds:
        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("experiment", {})["seed"] = int(seed)
        if multi:
            cfg["experiment"]["output_dir"] = str(base_out / f"seed_{seed}")
        seed_dirs.append(Path(cfg["experiment"]["output_dir"]))
        print(f"\n=== Running seed {seed} -> {cfg['experiment']['output_dir']} ===", flush=True)
        run_experiment(cfg, root=ROOT)

    if multi:
        aggregate = aggregate_seeds(seed_dirs, base_out / "aggregate.json")
        if aggregate:
            metrics = aggregate["metrics"]
            print("\n=== Multi-seed test metrics ===")
            for key, stats in metrics.items():
                print(f"{key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


if __name__ == "__main__":
    main()
