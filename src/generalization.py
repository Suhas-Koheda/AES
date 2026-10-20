"""Held-out-prompt generalization protocol.

Evaluation 2 is intentionally separate from the official benchmark.  For each
prompt, the prompt is removed from **both** training and validation; the model
is trained on the remaining prompts, selected on a validation split of those
prompts, and evaluated on every essay of the held-out prompt.

The official ASAP 2.0 test file contains only five of the seven training
prompts, so the held-out-prompt sweep uses the official *training* partition
as its pool.  The official test labels are never touched by this protocol.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .metrics import aggregate_seed_metrics, to_plain_floats
from .utils import ensure_dir, get_logger, save_json

logger = get_logger()


def run_heldout_prompt_sweep(
    cfg: dict,
    prompts: Iterable[str],
    output_root: str | Path,
    runner=None,
) -> dict:
    """Run one experiment per held-out prompt and aggregate the results."""
    if runner is None:
        from .runner import run_experiment as runner

    output_root = ensure_dir(output_root)
    per_prompt: Dict[str, dict] = {}
    summaries: List[dict] = []

    for prompt in prompts:
        logger.info("=" * 70)
        logger.info("Held-out prompt: %s", prompt)
        logger.info("=" * 70)
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("experiment", {})
        run_cfg["experiment"]["protocol"] = "heldout_prompt"
        run_cfg["experiment"]["heldout_prompt"] = prompt
        safe_name = _slug(prompt)
        run_cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'run')}__heldout_{safe_name}"
        run_cfg["experiment"]["output_dir"] = str(Path(output_root) / safe_name)

        summary_path = Path(run_cfg["experiment"]["output_dir"]) / "summary.json"
        if summary_path.exists():
            # A completed prompt survives session restarts; the sweep is
            # expensive (one full training run per prompt).
            logger.info("Reusing completed held-out run: %s", summary_path)
            with open(summary_path, encoding="utf-8") as fh:
                summary = json.load(fh)
            test = {
                "n": summary.get("test_n"),
                "qwk": summary.get("test_qwk"),
                "mae": summary.get("test_mae"),
                "rmse": summary.get("test_rmse"),
                "accuracy": summary.get("test_accuracy"),
                "spearman": summary.get("test_spearman"),
            }
            best_epoch = summary.get("best_epoch")
            best_val_qwk = summary.get("best_val_qwk")
        else:
            results = runner(run_cfg)
            test = results["test"]["overall"]
            best_epoch = results["training"].get("best_epoch")
            best_val_qwk = results["training"].get("best_metric")

        per_prompt[prompt] = {
            "n": test["n"],
            "qwk": test["qwk"],
            "mae": test["mae"],
            "rmse": test["rmse"],
            "accuracy": test["accuracy"],
            "spearman": test["spearman"],
            "best_epoch": best_epoch,
            "best_val_qwk": best_val_qwk,
            "output_dir": str(run_cfg["experiment"]["output_dir"]),
        }
        summaries.append(per_prompt[prompt])

    aggregate = _aggregate(summaries)
    payload = {
        "protocol": "heldout_prompt",
        "pool": "official_train_partition",
        "per_prompt": per_prompt,
        "aggregate": aggregate,
    }
    save_json(payload, Path(output_root) / "heldout_prompt_results.json")

    table = pd.DataFrame(
        [
            {"prompt": prompt, **{k: v for k, v in values.items() if k != "output_dir"}}
            for prompt, values in per_prompt.items()
        ]
    ).sort_values("prompt")
    table.to_csv(Path(output_root) / "heldout_prompt_results.csv", index=False)
    return payload


def _aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {}
    keys = ["qwk", "mae", "rmse", "accuracy", "spearman"]
    out = {"n_prompts": len(rows)}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        out[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "n_defined": int(finite.size),
        }
    return to_plain_floats(out)


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
