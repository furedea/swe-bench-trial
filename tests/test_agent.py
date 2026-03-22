"""Tests for mini-swe-agent invocation and patch collection."""

import subprocess
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import agent
import dataset


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Return tmp_path initialized as a git repo with one initial commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_collect_patch_returns_diff_string(git_repo: Path) -> None:
    (git_repo / "file.py").write_text("x = 2\n")

    result = agent.collect_patch(git_repo)

    assert "diff --git" in result
    assert "x = 2" in result


def test_collect_patch_returns_empty_when_no_changes(git_repo: Path) -> None:
    result = agent.collect_patch(git_repo)

    assert result == ""


def test_run_agent_calls_mini_with_correct_args(tmp_path: Path, mocker: MockerFixture) -> None:
    mock_run = mocker.patch("agent.subprocess.run")
    task = dataset.SWETask(tmp_path, "Fix the bug in units", "claude-sonnet-4-6")

    agent.run_agent(task)

    mock_run.assert_called_once_with(
        [
            "mini",
            "--model",
            "anthropic/claude-sonnet-4-6",
            "--task",
            "Fix the bug in units",
            "--yolo",
            "--exit-immediately",
        ],
        cwd=tmp_path,
        check=True,
    )
