"""Tests for one-shot prompt construction."""

from pathlib import Path

import pytest

import prompt
import retrieval


@pytest.fixture
def single_file_prompt() -> str:
    files = (retrieval.RetrievedFile(Path("units.py"), "class Unit: pass\n"),)
    return prompt.build_prompt("Fix the unit bug", files)


def test_build_prompt_contains_problem_statement(single_file_prompt: str) -> None:
    assert "Fix the unit bug" in single_file_prompt


def test_build_prompt_contains_file_path_and_content(single_file_prompt: str) -> None:
    assert "units.py" in single_file_prompt
    assert "class Unit: pass" in single_file_prompt


def test_build_prompt_contains_multiple_files() -> None:
    files = (
        retrieval.RetrievedFile(Path("a.py"), "x = 1\n"),
        retrieval.RetrievedFile(Path("b.py"), "y = 2\n"),
    )

    result = prompt.build_prompt("Fix something", files)

    assert "a.py" in result
    assert "b.py" in result


def test_build_prompt_with_no_files() -> None:
    result = prompt.build_prompt("Fix the bug", ())

    assert "Fix the bug" in result
    assert isinstance(result, str)
