"""Tests for dataset loading and repository setup."""

from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

import dataset


def test_swe_instance_from_dict_returns_correct_fields() -> None:
    row: dict[str, Any] = {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "base_commit": "abc123def456abc123def456abc123def456abc1",
        "problem_statement": "Fix the bug in units",
    }
    instance = dataset.SWEInstance.from_dict(row)
    assert instance.instance_id == "astropy__astropy-12907"
    assert instance.repo == "astropy/astropy"
    assert instance.base_commit == "abc123def456abc123def456abc123def456abc1"
    assert instance.problem_statement == "Fix the bug in units"


def test_swe_instance_from_dict_raises_on_missing_key() -> None:
    row: dict[str, Any] = {"instance_id": "foo"}
    with pytest.raises(KeyError):
        dataset.SWEInstance.from_dict(row)


def test_load_instance_returns_matching_instance(mocker: MockerFixture) -> None:
    mock_row: dict[str, Any] = {
        "instance_id": dataset.DEFAULT_INSTANCE_ID,
        "repo": "astropy/astropy",
        "base_commit": "abc123def456abc123def456abc123def456abc1",
        "problem_statement": "Fix the bug",
    }
    mock_load = mocker.patch("dataset.datasets.load_dataset", return_value=[mock_row])

    instance = dataset.load_instance(dataset.DEFAULT_INSTANCE_ID)

    assert instance.instance_id == dataset.DEFAULT_INSTANCE_ID
    mock_load.assert_called_once_with("princeton-nlp/SWE-bench_Lite", split="test")


def test_load_instance_raises_on_unknown_id(mocker: MockerFixture) -> None:
    mocker.patch("dataset.datasets.load_dataset", return_value=[])
    with pytest.raises(ValueError, match="Instance not found"):
        dataset.load_instance("nonexistent__repo-0")


def test_setup_repo_clones_and_resets(tmp_path: Path, mocker: MockerFixture) -> None:
    instance = dataset.SWEInstance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="abc123",
        problem_statement="Fix the bug",
    )
    mock_run = mocker.patch("dataset.subprocess.run")

    dataset.setup_repo(instance, tmp_path)

    expected_path = tmp_path / "astropy__astropy-12907"
    assert mock_run.call_count == 3
    mock_run.assert_any_call(
        ["git", "clone", "https://github.com/astropy/astropy.git", str(expected_path)],
        check=True,
    )
    mock_run.assert_any_call(
        ["git", "reset", "--hard", "abc123"],
        cwd=expected_path,
        check=True,
    )
    mock_run.assert_any_call(
        ["git", "clean", "-fd"],
        cwd=expected_path,
        check=True,
    )


def test_setup_repo_skips_clone_if_exists(tmp_path: Path, mocker: MockerFixture) -> None:
    instance = dataset.SWEInstance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="abc123",
        problem_statement="Fix the bug",
    )
    existing = tmp_path / "astropy__astropy-12907"
    existing.mkdir()
    mock_run = mocker.patch("dataset.subprocess.run")

    dataset.setup_repo(instance, tmp_path)

    # clone is skipped, but reset + clean always run
    assert mock_run.call_count == 2


def test_swe_task_is_frozen() -> None:
    task = dataset.SWETask(Path("/repo"), "Fix bug", "claude-sonnet-4-6")

    with pytest.raises(AttributeError):
        task.model_name = "other"  # type: ignore[misc]
