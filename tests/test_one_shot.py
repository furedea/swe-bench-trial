"""Tests for one-shot inference pipeline."""

from pathlib import Path

from pytest_mock import MockerFixture

import one_shot
import retrieval


def _mock_files(tmp_path: Path) -> tuple[retrieval.RetrievedFile, ...]:
    return (retrieval.RetrievedFile(tmp_path / "fix.py", "x = 1"),)


def test_run_one_shot_calls_correct_model_name(tmp_path: Path, mocker: MockerFixture) -> None:
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = "diff --git a/fix.py b/fix.py\n"
    mocker.patch("one_shot.retrieval.retrieve_files", return_value=_mock_files(tmp_path))
    mocker.patch("one_shot.prompt.build_prompt", return_value="Fix the bug")
    mock_completion = mocker.patch("one_shot.litellm.completion", return_value=mock_response)

    one_shot.run_one_shot(tmp_path, "Fix the bug", "claude-sonnet-4-6", top_k=3)

    mock_completion.assert_called_once_with(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Fix the bug"}],
    )


def test_run_one_shot_returns_extracted_diff(tmp_path: Path, mocker: MockerFixture) -> None:
    diff = "diff --git a/fix.py b/fix.py\n--- a/fix.py\n+++ b/fix.py\n"
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = f"Here is the fix:\n```diff\n{diff}```"
    mocker.patch("one_shot.retrieval.retrieve_files", return_value=_mock_files(tmp_path))
    mocker.patch("one_shot.prompt.build_prompt", return_value="prompt")
    mocker.patch("one_shot.litellm.completion", return_value=mock_response)

    result = one_shot.run_one_shot(tmp_path, "Fix the bug", "claude-sonnet-4-6", top_k=3)

    assert "diff --git" in result


def test_run_one_shot_returns_empty_string_when_content_is_none(tmp_path: Path, mocker: MockerFixture) -> None:
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = None
    mocker.patch("one_shot.retrieval.retrieve_files", return_value=_mock_files(tmp_path))
    mocker.patch("one_shot.prompt.build_prompt", return_value="prompt")
    mocker.patch("one_shot.litellm.completion", return_value=mock_response)

    result = one_shot.run_one_shot(tmp_path, "Fix the bug", "claude-sonnet-4-6", top_k=3)

    assert result == ""
