"""Tests for dataset loading and repository setup."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from dataset import DEFAULT_INSTANCE_ID, SWEInstance, load_instance, setup_repo


def test_swe_instance_from_dict_returns_correct_fields() -> None:
    row: dict[str, Any] = {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "base_commit": "abc123def456abc123def456abc123def456abc1",
        "problem_statement": "Fix the bug in units",
    }
    instance = SWEInstance.from_dict(row)
    assert instance.instance_id == "astropy__astropy-12907"
    assert instance.repo == "astropy/astropy"
    assert instance.base_commit == "abc123def456abc123def456abc123def456abc1"
    assert instance.problem_statement == "Fix the bug in units"


def test_swe_instance_from_dict_raises_on_missing_key() -> None:
    row: dict[str, Any] = {"instance_id": "foo"}
    with pytest.raises(KeyError):
        SWEInstance.from_dict(row)


def test_load_instance_returns_matching_instance() -> None:
    mock_row: dict[str, Any] = {
        "instance_id": DEFAULT_INSTANCE_ID,
        "repo": "astropy/astropy",
        "base_commit": "abc123def456abc123def456abc123def456abc1",
        "problem_statement": "Fix the bug",
    }

    with patch("dataset.load_dataset") as mock_load:
        mock_load.return_value = [mock_row]
        instance = load_instance(DEFAULT_INSTANCE_ID)

    assert instance.instance_id == DEFAULT_INSTANCE_ID
    mock_load.assert_called_once_with("princeton-nlp/SWE-bench_Lite", split="test")


def test_load_instance_raises_on_unknown_id() -> None:
    with patch("dataset.load_dataset") as mock_load:
        mock_load.return_value = []
        with pytest.raises(ValueError, match="Instance not found"):
            load_instance("nonexistent__repo-0")


def test_setup_repo_clones_and_checks_out(tmp_path: Path) -> None:
    instance = SWEInstance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="abc123",
        problem_statement="Fix the bug",
    )

    with patch("dataset.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = setup_repo(instance, tmp_path)

    expected_path = tmp_path / "astropy__astropy-12907"
    assert result == expected_path
    assert mock_run.call_count == 2
    mock_run.assert_any_call(
        ["git", "clone", "https://github.com/astropy/astropy.git", str(expected_path)],
        check=True,
    )
    mock_run.assert_any_call(
        ["git", "checkout", "abc123"],
        cwd=expected_path,
        check=True,
    )


def test_setup_repo_skips_clone_if_exists(tmp_path: Path) -> None:
    instance = SWEInstance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="abc123",
        problem_statement="Fix the bug",
    )
    existing = tmp_path / "astropy__astropy-12907"
    existing.mkdir()

    with patch("dataset.subprocess.run") as mock_run:
        result = setup_repo(instance, tmp_path)

    assert result == existing
    mock_run.assert_not_called()
