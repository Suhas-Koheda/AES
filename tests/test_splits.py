"""Split protocol tests: stratification, determinism and prompt isolation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import (
    make_heldout_prompt_splits,
    make_official_splits,
    stratified_train_val_split,
)


def test_stratified_split_sizes_and_disjoint(synthetic_train_df):
    train, val = stratified_train_val_split(synthetic_train_df, val_fraction=0.2, seed=7)
    assert len(train) + len(val) == len(synthetic_train_df)
    assert len(val) == pytest.approx(0.2 * len(synthetic_train_df), abs=5)
    assert set(train["essay_id"]).isdisjoint(set(val["essay_id"]))


def test_stratified_split_is_deterministic(synthetic_train_df):
    train_a, val_a = stratified_train_val_split(synthetic_train_df, 0.2, seed=11)
    train_b, val_b = stratified_train_val_split(synthetic_train_df, 0.2, seed=11)
    assert train_a["essay_id"].tolist() == train_b["essay_id"].tolist()
    assert val_a["essay_id"].tolist() == val_b["essay_id"].tolist()


def test_stratified_split_keeps_every_prompt(synthetic_train_df):
    train, val = stratified_train_val_split(synthetic_train_df, 0.2, seed=3)
    assert set(train["prompt_name"]) == set(synthetic_train_df["prompt_name"])
    assert set(val["prompt_name"]) == set(synthetic_train_df["prompt_name"])


def test_stratified_split_invalid_fraction(synthetic_train_df):
    with pytest.raises(ValueError):
        stratified_train_val_split(synthetic_train_df, 0.0)
    with pytest.raises(ValueError):
        stratified_train_val_split(synthetic_train_df, 1.0)


def test_official_splits_no_leakage(synthetic_train_df, synthetic_test_df):
    train, val, test, manifest = make_official_splits(
        synthetic_train_df, synthetic_test_df, val_fraction=0.15, seed=5
    )
    assert len(train) + len(val) == len(synthetic_train_df)
    assert len(test) == len(synthetic_test_df)
    assert set(train["essay_id"]).isdisjoint(set(test["essay_id"]))
    assert manifest.protocol == "official"
    assert len(manifest.train_ids) == len(train)


def test_heldout_prompt_excluded_from_train_and_val(synthetic_train_df):
    heldout = "Prompt B"
    train, val, test, manifest = make_heldout_prompt_splits(
        synthetic_train_df, heldout, val_fraction=0.15, seed=5
    )
    assert heldout not in set(train["prompt_name"])
    assert heldout not in set(val["prompt_name"])
    assert set(test["prompt_name"]) == {heldout}
    assert len(test) == int((synthetic_train_df["prompt_name"] == heldout).sum())
    assert manifest.heldout_prompt == heldout
    assert set(train["essay_id"]).isdisjoint(set(test["essay_id"]))


def test_heldout_prompt_unknown_raises(synthetic_train_df):
    with pytest.raises(ValueError):
        make_heldout_prompt_splits(synthetic_train_df, "Does not exist")


def test_heldout_prompt_counts_preserved(synthetic_train_df):
    heldout = "Prompt A"
    train, val, test, _ = make_heldout_prompt_splits(synthetic_train_df, heldout, seed=1)
    remaining = len(synthetic_train_df) - len(test)
    assert len(train) + len(val) == remaining
