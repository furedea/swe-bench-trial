"""Tests for main orchestration and prediction saving."""

import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import main


def test_save_prediction_writes_valid_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"
    result = main.PatchResult("astropy__astropy-12907", "claude-sonnet-4-6", "diff --git a/fix.py")

    main.save_prediction(result, output_path)

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["instance_id"] == "astropy__astropy-12907"
    assert data["model_name_or_path"] == "claude-sonnet-4-6"
    assert data["model_patch"] == "diff --git a/fix.py"


def test_save_prediction_appends_on_multiple_calls(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"
    main.save_prediction(main.PatchResult("id-1", "claude-sonnet-4-6", "patch-1"), output_path)
    main.save_prediction(main.PatchResult("id-2", "claude-sonnet-4-6", "patch-2"), output_path)

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_patch_result_is_frozen() -> None:
    result = main.PatchResult("id-1", "model", "patch")

    with pytest.raises(AttributeError):
        result.patch = "other"  # type: ignore[misc]


def test_main_raises_if_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main.main()


def test_main_agent_mode_raises_if_patch_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("sys.argv", ["main.py"])
    mock_instance = mocker.MagicMock()
    mock_instance.instance_id = "astropy__astropy-12907"
    mock_instance.problem_statement = "Fix the bug"
    mocker.patch("main.dataset.load_instance", return_value=mock_instance)
    mocker.patch("main.dataset.setup_repo")
    mocker.patch("main.agent.run_agent")
    mocker.patch("main.agent.collect_patch", return_value="")

    with pytest.raises(RuntimeError, match="Empty patch"):
        main.main()


def test_main_agent_mode_saves_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    output_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("sys.argv", ["main.py", "--output", str(output_path), "--workspace", str(tmp_path)])
    mock_instance = mocker.MagicMock()
    mock_instance.instance_id = "astropy__astropy-12907"
    mock_instance.problem_statement = "Fix the bug"
    mocker.patch("main.dataset.load_instance", return_value=mock_instance)
    mocker.patch("main.dataset.setup_repo")
    mocker.patch("main.agent.run_agent")
    mocker.patch("main.agent.collect_patch", return_value="diff --git a/fix.py")

    main.main()

    data = json.loads(output_path.read_text().strip())
    assert data["instance_id"] == "astropy__astropy-12907"
    assert data["model_patch"] == "diff --git a/fix.py"


def test_main_one_shot_mode_saves_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    output_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "one_shot", "--output", str(output_path)])
    mock_instance = mocker.MagicMock()
    mock_instance.instance_id = "astropy__astropy-12907"
    mock_instance.problem_statement = "Fix the bug"
    mocker.patch("main.dataset.load_instance", return_value=mock_instance)
    mocker.patch("main.dataset.setup_repo")
    mocker.patch("main.one_shot.run_one_shot", return_value="diff --git a/fix.py")

    main.main()

    data = json.loads(output_path.read_text().strip())
    assert data["instance_id"] == "astropy__astropy-12907"
    assert data["model_patch"] == "diff --git a/fix.py"
