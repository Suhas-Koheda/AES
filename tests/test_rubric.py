"""Rubric loader tests."""

from __future__ import annotations

import pytest

from src.rubric_loader import (
    RubricLoader,
    build_rubric_text,
    parse_score_descriptions,
    read_rubric_text,
)


def test_parse_six_bands(rubric_file):
    text = read_rubric_text(rubric_file)
    descriptions = parse_score_descriptions(text)
    assert set(descriptions) == {1, 2, 3, 4, 5, 6}
    assert "mastery" in descriptions[6].lower()


def test_build_rubric_text_order(rubric_file):
    loader = RubricLoader(rubric_file)
    full = loader.full_text
    positions = [full.index(f"SCORE OF {k}:") for k in (6, 5, 4, 3, 2, 1)]
    assert positions == sorted(positions)


def test_missing_band_raises():
    with pytest.raises(ValueError, match="missing score bands"):
        parse_score_descriptions("SCORE OF 6: only one band here")


def test_no_headers_raises():
    with pytest.raises(ValueError, match="could not find"):
        parse_score_descriptions("A rubric with no recognisable headers.")


def test_loader_for_prompt_is_shared(rubric_file):
    loader = RubricLoader(rubric_file)
    assert loader.for_prompt("Prompt A") == loader.for_prompt("Prompt B") == loader.full_text
    assert loader.describe()["shared_across_prompts"] is True


def test_loader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_rubric_text(tmp_path / "missing.txt")


def test_build_rubric_text_roundtrip(rubric_file):
    loader = RubricLoader(rubric_file)
    rebuilt = build_rubric_text(loader.descriptions)
    assert rebuilt == loader.full_text
