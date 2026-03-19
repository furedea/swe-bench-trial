"""Tests for main orchestration and prediction saving."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import save_prediction, main


def test_save_prediction_writes_valid_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"
    save_prediction(
        instance_id="astropy__astropy-12907",
        model="claude-sonnet-4-6",
        patch="diff --git a/fix.py",
        output_path=output_path,
    )

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["instance_id"] == "astropy__astropy-12907"
    assert data["model_name_or_path"] == "claude-sonnet-4-6"
    assert data["model_patch"] == "diff --git a/fix.py"


def test_save_prediction_appends_on_multiple_calls(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"
    save_prediction("id-1", "claude-sonnet-4-6", "patch-1", output_path)
    save_prediction("id-2", "claude-sonnet-4-6", "patch-2", output_path)

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_main_raises_if_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main()


def test_main_raises_if_patch_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    mock_instance = MagicMock()
    mock_instance.instance_id = "astropy__astropy-12907"
    mock_instance.problem_statement = "Fix the bug"

    with (
        patch("main.load_instance", return_value=mock_instance),
        patch("main.setup_repo", return_value=tmp_path),
        patch("main.run_agent", return_value=""),
    ):
        with pytest.raises(RuntimeError, match="Empty patch"):
            main()


def test_main_saves_prediction_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    mock_instance = MagicMock()
    mock_instance.instance_id = "astropy__astropy-12907"
    mock_instance.problem_statement = "Fix the bug"
    output_path = tmp_path / "predictions.jsonl"

    with (
        patch("main.load_instance", return_value=mock_instance),
        patch("main.setup_repo", return_value=tmp_path),
        patch("main.run_agent", return_value="diff --git a/fix.py"),
        patch("main.OUTPUT_PATH", output_path),
        patch("main.WORKSPACE_DIR", tmp_path),
    ):
        main()

    data = json.loads(output_path.read_text().strip())
    assert data["instance_id"] == "astropy__astropy-12907"
    assert data["model_patch"] == "diff --git a/fix.py"
