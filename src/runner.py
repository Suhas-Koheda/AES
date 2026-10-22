"""Train and evaluate the single question/rubric cross-attention scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .collator import CollatorConfig, EssayCollator
from .data import (
    DEMOGRAPHIC_COLUMNS,
    EssayDataset,
    PromptGroupedBatchSampler,
    load_asap_data,
    make_heldout_prompt_splits,
    make_official_splits,
    save_split_manifest,
)
from .evaluate import (
    evaluate_predictions,
    predict,
    save_evaluation,
    summarise,
)
from .fairness import build_fairness_report, save_fairness_report
from .labels import NUM_CLASSES
from .losses import LossComputer, compute_class_weights
from .metrics import compute_metrics, to_plain_floats
from .models import build_model, describe_model
from .rubric_loader import load_rubric
from .train import Trainer
from .utils import (
    PROJECT_ROOT,
    deep_get,
    describe_environment,
    ensure_dir,
    get_device,
    get_logger,
    progress,
    resolve_path,
    save_json,
    set_seed,
)

logger = get_logger()


# --------------------------------------------------------------------------- #
# Paths & small helpers
# --------------------------------------------------------------------------- #
def _resolve(path: Optional[str], root: Path) -> Optional[Path]:
    if path is None:
        return None
    return resolve_path(path, base=root)


def _experiment_dir(cfg: dict, root: Path) -> Path:
    out = deep_get(cfg, "experiment.output_dir", "results/run")
    return ensure_dir(_resolve(out, root))


def class_counts(df: pd.DataFrame) -> list[int]:
    counts = df["score"].astype(int).value_counts().sort_index()
    return [int(counts.get(score, 0)) for score in range(1, NUM_CLASSES + 1)]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_and_split(cfg: dict, root: Path):
    data_cfg = cfg.get("data", {})
    protocol = deep_get(cfg, "experiment.protocol", "official")
    seed = int(deep_get(cfg, "experiment.seed", 42))
    val_fraction = float(data_cfg.get("val_fraction", 0.15))

    if data_cfg.get("dataset") == "asap1":
        from .asap1 import load_asap1, make_prompt_fold

        asap1 = load_asap1(
            _resolve(data_cfg.get("data_path"), root),
            _resolve(data_cfg.get("contexts_path"), root),
        )
        prompt_id = int(deep_get(cfg, "experiment.prompt_id"))
        fold = int(deep_get(cfg, "experiment.fold"))
        train_split, val_split, test_split = make_prompt_fold(asap1, prompt_id, fold, seed)
        manifest = type("ASAP1Manifest", (), {
            "protocol": "prompt_specific_5fold_60_20_20",
            "summary": lambda self, tr, va, te: {
                "protocol": self.protocol, "seed": seed, "prompt_id": prompt_id,
                "fold": fold, "train_n": len(tr), "val_n": len(va), "test_n": len(te),
            },
        })()
        logger.info(
            "ASAP 1.0 prompt %d fold %d: train=%d val=%d test=%d",
            prompt_id, fold, len(train_split), len(val_split), len(test_split),
        )
        return train_split, val_split, test_split, manifest, None

    train_csv = _resolve(data_cfg.get("train_csv"), root)
    test_csv = _resolve(data_cfg.get("test_csv"), root)
    train_df, test_df = load_asap_data(train_csv, test_csv)

    if protocol == "official":
        if test_df is None:
            raise ValueError("official protocol requires data.test_csv")
        train_split, val_split, test_split, manifest = make_official_splits(
            train_df, test_df, val_fraction=val_fraction, seed=seed
        )
    elif protocol == "heldout_prompt":
        heldout = deep_get(cfg, "experiment.heldout_prompt")
        if not heldout:
            raise ValueError("heldout_prompt protocol requires experiment.heldout_prompt")
        # Use the official training partition as the pool so the official test
        # labels never influence the held-out-prompt experiment.
        train_split, val_split, test_split, manifest = make_heldout_prompt_splits(
            train_df, heldout, val_fraction=val_fraction, seed=seed
        )
    else:
        raise ValueError(f"unknown protocol: {protocol!r}")

    logger.info(
        "Split [%s] train=%d val=%d test=%d | prompts train=%s test=%s",
        protocol,
        len(train_split),
        len(val_split),
        len(test_split),
        sorted(train_split["prompt_name"].unique()),
        sorted(test_split["prompt_name"].unique()),
    )

    # Optional subsampling for smoke tests / debugging (never used for reported
    # experiments unless explicitly set in the config).
    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_eval_samples")
    max_test = data_cfg.get("max_eval_samples")
    if max_train:
        train_split = train_split.sample(
            n=min(int(max_train), len(train_split)), random_state=seed
        ).reset_index(drop=True)
    if max_val:
        val_split = val_split.sample(
            n=min(int(max_val), len(val_split)), random_state=seed
        ).reset_index(drop=True)
    if max_test:
        test_split = test_split.sample(
            n=min(int(max_test), len(test_split)), random_state=seed
        ).reset_index(drop=True)

    return train_split, val_split, test_split, manifest, test_df


def build_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    model_name = deep_get(cfg, "model.pretrained_model", "microsoft/deberta-v3-small")
    cache_dir = deep_get(cfg, "model.cache_dir")
    try:
        # Fast tokenizer is picklable across DataLoader workers and implements
        # build_inputs_with_special_tokens; prefer it when available.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, cache_dir=cache_dir
        )
    except (ValueError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, cache_dir=cache_dir
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def build_collator(cfg: dict, tokenizer) -> EssayCollator:
    data_cfg = cfg.get("data", {})
    max_answer = int(data_cfg.get("max_answer_length", 512))
    max_context = int(data_cfg.get("max_context_length", 512))
    collator_cfg = CollatorConfig(
        input_mode=data_cfg.get("input_mode", "qra_cross"),
        max_answer_length=max_answer,
        max_context_length=max_context,
        truncation=data_cfg.get("truncation", "head_tail"),
        head_ratio=float(data_cfg.get("head_ratio", 0.5)),
        chunk=bool(data_cfg.get("chunk", False)),
        chunk_size=data_cfg.get("chunk_size"),
        chunk_overlap=int(data_cfg.get("chunk_overlap", 32)),
        max_chunks=int(data_cfg.get("max_chunks", 4)),
    )
    return EssayCollator(tokenizer, collator_cfg)


def make_loader(
    dataset: EssayDataset,
    collator: EssayCollator,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
    prompt_grouped: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _init_worker(worker_id: int):
        np.random.seed(seed + worker_id)

    common = dict(
        dataset=dataset,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=_init_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    if prompt_grouped:
        batch_sampler = PromptGroupedBatchSampler(
            dataset.df["prompt_name"].astype(str).tolist(),
            batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(batch_sampler=batch_sampler, **common)

    return DataLoader(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **common,
    )


# --------------------------------------------------------------------------- #
# Neural path
# --------------------------------------------------------------------------- #
def run_neural(cfg: dict, root: Path, out_dir: Path) -> dict:
    seed = int(deep_get(cfg, "experiment.seed", 42))
    set_seed(seed)
    device = get_device(deep_get(cfg, "train.device", "auto"))
    logger.info("Device: %s", device)

    if device.type == "cpu" and not deep_get(cfg, "train.allow_cpu", False):
        logger.warning(
            "Running neural training on CPU. This is supported but slow; set "
            "train.allow_cpu=true to silence this warning, or use a GPU runtime."
        )

    train_df, val_df, test_df, manifest, _ = load_and_split(cfg, root)
    save_split_manifest(train_df, val_df, test_df, manifest, out_dir)

    if deep_get(cfg, "data.dataset") == "asap1":
        rubric_text = ""
        save_json({
            "source": "prompt-specific text supplied in configured ASAP 1.0 context JSON",
            "rubric_invented": False,
        }, out_dir / "rubric.json")
    else:
        rubric = load_rubric(_resolve(deep_get(cfg, "data.rubric_path"), root))
        rubric_text = rubric.full_text
        save_json(rubric.describe(), out_dir / "rubric.json")

    tokenizer = build_tokenizer(cfg)
    collator = build_collator(cfg, tokenizer)

    train_ds = EssayDataset(train_df, rubric_text)
    val_ds = EssayDataset(val_df, rubric_text)
    test_ds = EssayDataset(test_df, rubric_text)

    num_workers = int(deep_get(cfg, "data.num_workers", 2))
    batch_size = int(deep_get(cfg, "train.batch_size", 16))
    eval_batch_size = int(deep_get(cfg, "train.eval_batch_size", batch_size * 2))
    # For cross-attention the context is constant per prompt, so grouping
    # batches by prompt lets the model encode it once per batch.
    prompt_grouped = deep_get(cfg, "data.input_mode", "") == "qra_cross"
    train_loader = make_loader(
        train_ds, collator, batch_size, True, num_workers, seed,
        prompt_grouped=prompt_grouped,
    )
    val_loader = make_loader(
        val_ds, collator, eval_batch_size, False, num_workers, seed,
        prompt_grouped=prompt_grouped,
    )
    test_loader = make_loader(
        test_ds, collator, eval_batch_size, False, num_workers, seed,
        prompt_grouped=prompt_grouped,
    )

    model = build_model(cfg.get("model", {}), cache_dir=deep_get(cfg, "model.cache_dir"))
    model_info = describe_model(model)
    logger.info("Model: %s", json.dumps(model_info))
    save_json(model_info, out_dir / "model.json")


    class_weight_mode = deep_get(cfg, "train.class_weight", "none")
    weights = compute_class_weights(
        class_counts(train_df), NUM_CLASSES, class_weight_mode
    )
    if weights is not None:
        logger.info("Class weights: %s", weights.tolist())

    loss_computer = LossComputer(
        class_weights=weights,
        label_smoothing=float(deep_get(cfg, "train.label_smoothing", 0.0)),
    )

    trainer = Trainer(
        model,
        device=device,
        loss_computer=loss_computer,
        decode_mode=deep_get(cfg, "eval.decode_mode", "argmax"),
        out_dir=out_dir,
        learning_rate=float(deep_get(cfg, "train.learning_rate", 2e-5)),
        weight_decay=float(deep_get(cfg, "train.weight_decay", 0.01)),
        warmup_ratio=float(deep_get(cfg, "train.warmup_ratio", 0.1)),
        epochs=int(deep_get(cfg, "train.epochs", 3)),
        grad_accumulation=int(deep_get(cfg, "train.grad_accumulation", 1)),
        max_grad_norm=float(deep_get(cfg, "train.max_grad_norm", 1.0)),
        fp16=bool(deep_get(cfg, "train.fp16", True)),
        selection_metric=deep_get(cfg, "train.selection_metric", "qwk"),
        early_stopping_patience=int(deep_get(cfg, "train.early_stopping_patience", 2)),
        scheduler=deep_get(cfg, "train.scheduler", "linear"),
        log_every=int(deep_get(cfg, "train.log_every", 50)),
        max_train_steps=deep_get(cfg, "train.max_train_steps"),
        save_total_limit=int(deep_get(cfg, "train.save_total_limit", 2)),
    )

    resume_from = deep_get(cfg, "train.resume_from")
    train_summary = trainer.fit(train_loader, val_loader, resume_from=resume_from)

    # ---- Final evaluation with the best checkpoint (selected on val QWK) ----
    best_path = out_dir / "checkpoint_best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded best checkpoint from epoch %s", state.get("epoch"))
    model.to(device)

    results = {
        "experiment": deep_get(cfg, "experiment.name", "neural"),
        "protocol": deep_get(cfg, "experiment.protocol", "official"),
        "heldout_prompt": deep_get(cfg, "experiment.heldout_prompt"),
        "model": model_info,
        "training": train_summary,
        "environment": describe_environment(),
        "config": cfg,
    }

    group_columns = deep_get(cfg, "eval.group_columns", [])
    for split_name, loader, df in (
        ("val", val_loader, val_df),
        ("test", test_loader, test_df),
    ):
        preds = predict(
            model,
            loader,
            device,
            decode_mode=deep_get(cfg, "eval.decode_mode", "argmax"),
            use_amp=bool(deep_get(cfg, "train.fp16", True)),
            desc=f"eval {split_name}",
        )
        if deep_get(cfg, "data.dataset") == "asap1":
            from .asap1 import restore_native_scores

            # `predict` decodes the unchanged six-class head on its training
            # scale. Report both truth and predictions on the native prompt
            # score scale using the predeclared prompt bounds.
            preds["true_score"] = df["raw_score"].astype(int).to_numpy()
            # Decode positionally so each example uses its prompt's native
            # score bounds.
            preds["predicted_score"] = np.asarray([
                restore_native_scores(int(prompt), [cls])[0]
                for prompt, cls in zip(preds["prompt_names"], preds["predicted_class"])
            ])
        metadata = df[[c for c in ["essay_id", "pubpriv"] + group_columns if c in df.columns]].copy()
        evaluation = evaluate_predictions(preds, metadata=metadata, group_columns=group_columns)
        save_evaluation(preds, evaluation, out_dir, split_name, metadata=metadata)
        results[split_name] = evaluation
        logger.info("[%s] %s", split_name, summarise(evaluation))

        if split_name == "test":
            fairness = build_fairness_report(preds, df, group_columns)
            save_fairness_report(fairness, out_dir / "fairness")

    save_json(results, out_dir / "results.json")

    summary = {
        "experiment": results["experiment"],
        "protocol": results["protocol"],
        "heldout_prompt": results["heldout_prompt"],
        "seed": seed,
        "model": model_info,
        "best_epoch": train_summary.get("best_epoch"),
        "best_val_qwk": train_summary.get("best_metric"),
        "val_qwk": results["val"]["overall"]["qwk"],
        "test_qwk": results["test"]["overall"]["qwk"],
        "test_mae": results["test"]["overall"]["mae"],
        "test_rmse": results["test"]["overall"]["rmse"],
        "test_accuracy": results["test"]["overall"]["accuracy"],
        "test_pearson": results["test"]["overall"]["pearson"],
        "test_spearman": results["test"]["overall"]["spearman"],
        "test_n": results["test"]["overall"]["n"],
    }
    save_json(summary, out_dir / "summary.json")
    logger.info("Summary: %s", json.dumps(to_plain_floats(summary)))

    if bool(deep_get(cfg, "train.cleanup_checkpoints_after_run", False)):
        removed = 0
        for checkpoint in out_dir.glob("checkpoint_*.pt"):
            checkpoint.unlink(missing_ok=True)
            removed += 1
        if removed:
            logger.info(
                "Removed %d checkpoint file(s) from %s after successful run",
                removed,
                out_dir,
            )

    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_experiment(cfg: dict, root: Optional[Path] = None) -> dict:
    root = Path(root) if root else PROJECT_ROOT
    out_dir = _experiment_dir(cfg, root)
    save_json(cfg, out_dir / "config.json")
    save_json(describe_environment(), out_dir / "environment.json")

    return run_neural(cfg, root, out_dir)
