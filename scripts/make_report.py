#!/usr/bin/env python
"""Aggregate results into comparison tables and figures.

Reads ``results/**/summary.json`` and ``metrics_test.json`` written by the
runner, then produces:

- ``results/tables/model_comparison.csv`` / ``.md``
- ``results/tables/per_prompt_test.csv``
- ``results/tables/ablation.csv`` / ``.md``
- ``results/figures/*.png`` (QWK comparison, per-prompt QWK, confusion matrix,
  score distributions)

Usage
-----
python scripts/make_report.py
python scripts/make_report.py --ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from configs_map import ABLATION_ROWS, OUTPUT_DIRS  # noqa: E402


def load_summary(directory: Path) -> Optional[dict]:
    path = directory / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["_dir"] = str(directory)
    return data


def load_metrics(directory: Path, split: str = "test") -> Optional[dict]:
    path = directory / "metrics" / f"metrics_{split}.json"
    if not path.exists():
        path = directory / f"metrics_{split}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect(results_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for summary_path in sorted(results_root.rglob("summary.json")):
        directory = summary_path.parent
        summary = load_summary(directory)
        if not summary:
            continue
        model_block = summary.get("model") or {}
        if not isinstance(model_block, dict):
            model_block = {}
        params = model_block.get("parameters")
        if isinstance(params, dict):
            params = params.get("total")
        rows.append(
            {
                "experiment": summary.get("experiment"),
                "protocol": summary.get("protocol", "official"),
                "heldout_prompt": summary.get("heldout_prompt"),
                "model": model_block.get("class") or model_block.get("name"),
                "parameters_total": params,
                "val_qwk": summary.get("val_qwk"),
                "test_qwk": summary.get("test_qwk"),
                "test_mae": summary.get("test_mae"),
                "test_rmse": summary.get("test_rmse"),
                "test_accuracy": summary.get("test_accuracy"),
                "test_spearman": summary.get("test_spearman"),
                "test_n": summary.get("test_n"),
                "best_epoch": summary.get("best_epoch"),
                "directory": str(directory.relative_to(ROOT)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["protocol", "experiment"]).reset_index(drop=True)


def per_prompt_table(directories: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for label, directory in directories.items():
        metrics = load_metrics(directory)
        if not metrics:
            continue
        for prompt, values in metrics.get("per_prompt", {}).items():
            rows.append(
                {
                    "experiment": label,
                    "prompt_name": prompt,
                    "n": values.get("n"),
                    "qwk": values.get("qwk"),
                    "mae": values.get("mae"),
                    "rmse": values.get("rmse"),
                    "accuracy": values.get("accuracy"),
                    "spearman": values.get("spearman"),
                }
            )
    return pd.DataFrame(rows)


def ablation_table(results_root: Path) -> pd.DataFrame:
    rows = []
    for key, label in ABLATION_ROWS:
        directory = results_root / OUTPUT_DIRS.get(key, key)
        summary = load_summary(directory)
        if not summary:
            continue
        rows.append(
            {
                "row": key,
                "ablation": label,
                "val_qwk": summary.get("val_qwk"),
                "test_qwk": summary.get("test_qwk"),
                "test_mae": summary.get("test_mae"),
                "test_rmse": summary.get("test_rmse"),
                "test_accuracy": summary.get("test_accuracy"),
                "test_spearman": summary.get("test_spearman"),
                "best_epoch": summary.get("best_epoch"),
            }
        )
    return pd.DataFrame(rows)


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results found._\n"
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda v: f"{v:.4f}" if pd.notna(v) else "nan"
            )
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    lines = [header, separator]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def make_figures(directories: Dict[str, Path], figures_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable ({exc}); skipping figures")
        return

    ensure_dir(figures_dir)

    # 1. QWK comparison
    labels, values = [], []
    for label, directory in directories.items():
        summary = load_summary(directory)
        if summary and summary.get("test_qwk") is not None:
            labels.append(label)
            values.append(summary["test_qwk"])
    if labels:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
        ax.bar(labels, values, color="#4C72B0")
        ax.set_ylabel("Test QWK")
        ax.set_ylim(min(0, min(values) - 0.05), 1.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Test Quadratic Weighted Kappa")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(figures_dir / "qwk_comparison.png", dpi=150)
        plt.close(fig)

    for label, directory in directories.items():
        metrics = load_metrics(directory)
        if not metrics:
            continue
        # 2. Confusion matrix
        cm = metrics["overall"].get("confusion_matrix")
        if cm:
            cm = np.asarray(cm)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted score")
            ax.set_ylabel("Human score")
            ax.set_title(f"Confusion matrix: {label}")
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels(range(1, 7))
            ax.set_yticklabels(range(1, 7))
            for i in range(6):
                for j in range(6):
                    if cm[i, j]:
                        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(figures_dir / f"confusion_{_slug(label)}.png", dpi=150)
            plt.close(fig)

        # 3. Score distribution
        pred_path = directory / "predictions_test.csv"
        if pred_path.exists():
            frame = pd.read_csv(pred_path)
            fig, ax = plt.subplots(figsize=(6, 4))
            scores = range(1, 7)
            human = [int((frame["true_score"] == s).sum()) for s in scores]
            pred = [int((frame["predicted_score"] == s).sum()) for s in scores]
            width = 0.4
            ax.bar([s - width / 2 for s in scores], human, width, label="Human")
            ax.bar([s + width / 2 for s in scores], pred, width, label="Predicted")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_title(f"Score distribution: {label}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / f"distribution_{_slug(label)}.png", dpi=150)
            plt.close(fig)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--ablation", action="store_true", help="also emit the ablation table")
    args = parser.parse_args()

    results_root = ROOT / args.results_root
    tables_dir = ensure_dir(results_root / "tables")
    figures_dir = ensure_dir(results_root / "figures")

    comparison = collect(results_root)
    if comparison.empty:
        print(f"No summary.json files found under {results_root}")
        return
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    (tables_dir / "model_comparison.md").write_text(_to_markdown(comparison), encoding="utf-8")

    directories = {
        row["experiment"]: ROOT / row["directory"] for _, row in comparison.iterrows()
    }
    per_prompt = per_prompt_table(directories)
    if not per_prompt.empty:
        per_prompt.to_csv(tables_dir / "per_prompt_test.csv", index=False)

    if args.ablation:
        table = ablation_table(results_root)
        if not table.empty:
            table.to_csv(tables_dir / "ablation.csv", index=False)
            (tables_dir / "ablation.md").write_text(_to_markdown(table), encoding="utf-8")
            print(_to_markdown(table))

    make_figures(directories, figures_dir)
    print(f"\nWrote tables to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")
    print("\n" + _to_markdown(comparison))


if __name__ == "__main__":
    main()
