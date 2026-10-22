"""ASAP 1.0 score mapping, context validation and split tests."""
import json

import numpy as np
import pandas as pd
import pytest

from src.asap1 import load_asap1, make_prompt_fold, restore_native_scores


def _asap_frame():
    rows = []
    for prompt, (low, high) in {
        1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
        5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60),
    }.items():
        for score in range(low, high + 1):
            for repeat in range(5):
                rows.append({"essay_id": f"{prompt}-{score}-{repeat}", "essay_set": prompt,
                             "essay": f"essay {prompt} {score} {repeat}", "domain1_score": score})
    return pd.DataFrame(rows)


def test_asap1_score_mapping_and_source_context(tmp_path):
    frame = _asap_frame()
    data = tmp_path / "asap.csv"
    ctx = tmp_path / "contexts.json"
    frame.to_csv(data, index=False)
    ctx.write_text(json.dumps({str(i): f"verbatim source context {i}" for i in range(1, 9)}))
    loaded = load_asap1(data, ctx)
    assert set(loaded.prompt_id) == set(range(1, 9))
    assert loaded.score.between(1, 6).all()
    assert loaded.loc[loaded.prompt_id == 1, "raw_score"].min() == 2
    assert loaded.loc[loaded.prompt_id == 8, "raw_score"].max() == 60
    assert loaded.loc[loaded.prompt_id == 8, "assignment"].iloc[0] == "verbatim source context 8"
    assert restore_native_scores(1, [0, 5]).tolist() == [2, 12]
    assert restore_native_scores(8, [0, 5]).tolist() == [0, 60]


def test_asap1_prompt_folds_are_stratified_disjoint_and_deterministic(tmp_path):
    frame = _asap_frame()
    data, ctx = tmp_path / "data.csv", tmp_path / "contexts.json"
    frame.to_csv(data, index=False)
    ctx.write_text(json.dumps({str(i): f"context {i}" for i in range(1, 9)}))
    loaded = load_asap1(data, ctx)
    a = make_prompt_fold(loaded, 2, 0, 77)
    b = make_prompt_fold(loaded, 2, 0, 77)
    train, val, test = a
    assert train.essay_id.tolist() == b[0].essay_id.tolist()
    assert len(train) == 3 * len(val) == 3 * len(test)
    assert set(train.prompt_id) == set(val.prompt_id) == set(test.prompt_id) == {2}
    assert set(train.essay_id).isdisjoint(val.essay_id)
    assert set(train.essay_id).isdisjoint(test.essay_id)
    assert set(val.essay_id).isdisjoint(test.essay_id)


def test_asap1_requires_context_and_valid_prompt_scores(tmp_path):
    frame = _asap_frame()
    data, ctx = tmp_path / "data.csv", tmp_path / "contexts.json"
    frame.to_csv(data, index=False)
    ctx.write_text("{}")
    with pytest.raises(ValueError, match="context file needs"):
        load_asap1(data, ctx)
    frame.loc[0, "domain1_score"] = 100
    frame.to_csv(data, index=False)
    ctx.write_text(json.dumps({str(i): f"source {i}" for i in range(1, 9)}))
    with pytest.raises(ValueError, match="published range"):
        load_asap1(data, ctx)
