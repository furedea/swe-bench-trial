"""Tests for one-shot inference pipeline."""

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import dataset
import one_shot
import retrieval


@pytest.fixture
def prompt_template(tmp_path: Path) -> Path:
    path = tmp_path / "template.txt"
    path.write_text("{problem_statement}\n{code}\nGenerate a patch.")
    return path


@pytest.fixture
def task(tmp_path: Path) -> dataset.SWETask:
    return dataset.SWETask(tmp_path, "Fix the bug", "claude-sonnet-4-6")


def _mock_files(tmp_path: Path) -> tuple[retrieval.RetrievedFile, ...]:
    return (retrieval.RetrievedFile(tmp_path / "fix.py", "x = 1"),)


def test_run_one_shot_calls_correct_model_name(
    task: dataset.SWETask, prompt_template: Path, mocker: MockerFixture
) -> None:
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = "diff --git a/fix.py b/fix.py\n"
    mocker.patch("one_shot.retrieval.retrieve_files", return_value=_mock_files(task.repo_path))
    mocker.patch("one_shot.prompt.build_prompt", return_value="Fix the bug")
    mock_completion = mocker.patch("one_shot.litellm.completion", return_value=mock_response)

    one_shot.run_one_shot(task, top_k=3, prompt_template=prompt_template)

    mock_completion.assert_called_once_with(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Fix the bug"}],
    )


def test_run_one_shot_returns_extracted_diff(
    task: dataset.SWETask, prompt_template: Path, mocker: MockerFixture
) -> None:
    diff = "diff --git a/fix.py b/fix.py\n--- a/fix.py\n+++ b/fix.py\n"
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = f"Here is the fix:\n```diff\n{diff}```"
    mocker.patch("one_shot.retrieval.retrieve_files", return_value=_mock_files(task.repo_path))
    mocker.patch("one_shot.prompt.build_prompt", return_value="prompt")
    mocker.patch("one_shot.litellm.completion", return_value=mock_response)

    result = one_shot.run_one_shot(task, top_k=3, prompt_template=prompt_template)

    assert "diff --git" in result


def test_run_one_shot_returns_empty_string_when_content_is_none(
    task: dataset.SWETask, prompt_template: Path, mocker: MockerFixture
) -> None:
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = None
    mocker.patch("one_shot.retrieval.retrieve_files", return_value=_mock_files(task.repo_path))
    mocker.patch("one_shot.prompt.build_prompt", return_value="prompt")
    mocker.patch("one_shot.litellm.completion", return_value=mock_response)

    result = one_shot.run_one_shot(task, top_k=3, prompt_template=prompt_template)

    assert result == ""


def test_extract_diff_from_patch_tags() -> None:
    content = "Here is the fix:\n<patch>\n--- a/f.py\n+++ b/f.py\n</patch>"

    assert one_shot.extract_diff(content) == "--- a/f.py\n+++ b/f.py"


def test_extract_diff_from_fenced_code_block() -> None:
    content = "Fix:\n```diff\n--- a/f.py\n+++ b/f.py\n```"

    assert one_shot.extract_diff(content) == "--- a/f.py\n+++ b/f.py"


def test_extract_diff_from_raw_diff() -> None:
    content = "Some explanation\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@"

    assert one_shot.extract_diff(content).startswith("--- a/f.py")


def test_extract_diff_returns_empty_string_when_no_diff() -> None:
    assert one_shot.extract_diff("No diff here") == ""


def test_extract_diff_patch_tags_take_priority_over_code_block() -> None:
    content = "<patch>\n--- a/real.py\n+++ b/real.py\n</patch>\n```diff\n--- a/fake.py\n```"

    result = one_shot.extract_diff(content)

    assert "real.py" in result
    assert "fake.py" not in result
