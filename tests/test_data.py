"""Dataset loading, schema validation and split-isolation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    EssayDataset,
    assert_no_leakage,
    check_prompt_disjointness,
    fill_missing_assignments,
    load_asap_data,
    make_heldout_prompt_splits,
    make_official_splits,
    stratified_train_val_split,
    verify_schema,
)
from src.labels import NUM_CLASSES


def test_make_synthetic_df_has_required_columns(synthetic_train_df):
    for column in ["essay_id", "score", "full_text", "assignment", "prompt_name"]:
        assert column in synthetic_train_df.columns
    assert synthetic_train_df["score"].between(1, 6).all()


def test_verify_schema_ok(synthetic_train_df):
    summary = verify_schema(synthetic_train_df, "synthetic")
    assert summary["n_rows"] == len(synthetic_train_df)
    assert summary["n_prompts"] == 3


def test_verify_schema_missing_column(synthetic_train_df):
    with pytest.raises(ValueError, match="missing required columns"):
        verify_schema(synthetic_train_df.drop(columns=["full_text"]), "broken")


def test_verify_schema_bad_score(synthetic_train_df):
    broken = synthetic_train_df.copy()
    broken.loc[0, "score"] = 9
    with pytest.raises(ValueError, match="scores outside"):
        verify_schema(broken, "broken")


def test_verify_schema_duplicate_ids(synthetic_train_df):
    broken = synthetic_train_df.copy()
    broken.loc[1, "essay_id"] = broken.loc[0, "essay_id"]
    with pytest.raises(ValueError, match="duplicated essay_id"):
        verify_schema(broken, "broken")


def test_fill_missing_assignment_from_prompt_level_text(synthetic_train_df):
    broken = synthetic_train_df.copy()
    prompt = "Prompt A"
    expected = broken.loc[broken["prompt_name"] == prompt, "assignment"].iloc[0]
    broken.loc[broken["prompt_name"] == prompt, "assignment"] = None
    with pytest.raises(ValueError, match="null assignment"):
        verify_schema(broken, "broken")

    filled, n_filled = fill_missing_assignments(broken, "broken")
    assert n_filled == int((broken["prompt_name"] == prompt).sum())
    assert filled["assignment"].isna().sum() == 0
    assert (filled.loc[filled["prompt_name"] == prompt, "assignment"] == expected).all()
    verify_schema(filled, "filled")


def test_fill_missing_assignment_no_source_raises():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "essay_id": ["a", "b"],
            "score": [1, 2],
            "full_text": ["x", "y"],
            "set": ["train", "train"],
            "pubpriv": [0, 0],
            "assignment": [None, None],
            "prompt_name": ["P", "P"],
        }
    )
    with pytest.raises(ValueError, match="no non-null example"):
        fill_missing_assignments(frame, "broken")


def test_load_asap_data(tmp_path, synthetic_train_df, synthetic_test_df):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    synthetic_train_df.to_csv(train_path, index=False)
    synthetic_test_df.to_csv(test_path, index=False)

    train, test = load_asap_data(train_path, test_path)
    assert len(train) == len(synthetic_train_df)
    assert "label" in train.columns
    assert (train["label"] == train["score"] - 1).all()
    assert train["label"].between(0, NUM_CLASSES - 1).all()


def test_load_asap_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_asap_data(tmp_path / "nope.csv")


def test_essay_dataset_fields(synthetic_train_df, rubric_file):
    rubric = rubric_file.read_text()
    dataset = EssayDataset(synthetic_train_df.iloc[:5], rubric)
    sample = dataset[0]
    assert set(["essay_id", "question", "answer", "rubric", "label", "score", "prompt_name"]).issubset(sample)
    assert sample["label"] == sample["score"] - 1
    assert 0 <= sample["label"] <= 5
    assert len(dataset) == 5


def test_assert_no_leakage_detects_overlap(synthetic_train_df):
    a = synthetic_train_df.iloc[:10].copy()
    b = synthetic_train_df.iloc[5:15].copy()
    with pytest.raises(AssertionError, match="data leakage"):
        assert_no_leakage(a, b)


def test_check_prompt_disjointness(synthetic_train_df):
    prompt = synthetic_train_df["prompt_name"].iloc[0]
    with pytest.raises(AssertionError):
        check_prompt_disjointness(synthetic_train_df, synthetic_train_df, prompt)
