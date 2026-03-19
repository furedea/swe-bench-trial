"""Tests for mini-swe-agent invocation and patch collection."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runner import collect_patch, run_agent


def test_collect_patch_returns_diff_string(tmp_path: Path) -> None:
    # Init a real git repo, modify a file, then collect patch
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    target = tmp_path / "fix.py"
    target.write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    target.write_text("x = 2\n")

    patch = collect_patch(tmp_path)

    assert "diff --git" in patch
    assert "x = 2" in patch


def test_collect_patch_returns_empty_when_no_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    patch = collect_patch(tmp_path)

    assert patch == ""


def test_run_agent_calls_mini_with_correct_args(tmp_path: Path) -> None:
    with patch("runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch("runner.collect_patch", return_value="diff --git a/fix.py"):
            result = run_agent(tmp_path, "Fix the bug in units", "claude-sonnet-4-6")

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
    assert result == "diff --git a/fix.py"
