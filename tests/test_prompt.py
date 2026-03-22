"""Tests for one-shot prompt construction."""

from pathlib import Path

import pytest

import prompt
import retrieval


@pytest.fixture
def template(tmp_path: Path) -> Path:
    path = tmp_path / "template.txt"
    path.write_text("<issue>\n{problem_statement}\n</issue>\n<code>\n{code}\n</code>\nGenerate a patch.")
    return path


def test_build_prompt_contains_problem_statement(template: Path) -> None:
    files = (retrieval.RetrievedFile(Path("units.py"), "class Unit: pass\n"),)

    result = prompt.build_prompt("Fix the unit bug", files, template)

    assert "Fix the unit bug" in result


def test_build_prompt_contains_file_path_and_content(template: Path) -> None:
    files = (retrieval.RetrievedFile(Path("units.py"), "class Unit: pass\n"),)

    result = prompt.build_prompt("Fix the unit bug", files, template)

    assert "units.py" in result
    assert "class Unit: pass" in result


def test_build_prompt_contains_multiple_files(template: Path) -> None:
    files = (
        retrieval.RetrievedFile(Path("a.py"), "x = 1\n"),
        retrieval.RetrievedFile(Path("b.py"), "y = 2\n"),
    )

    result = prompt.build_prompt("Fix something", files, template)

    assert "a.py" in result
    assert "b.py" in result


def test_build_prompt_with_no_files(template: Path) -> None:
    result = prompt.build_prompt("Fix the bug", (), template)

    assert "Fix the bug" in result
    assert isinstance(result, str)


def test_format_file_with_lines_adds_line_numbers() -> None:
    content = "alpha\nbeta\ngamma"

    result = prompt._format_file_with_lines("foo.py", content)

    assert "[start of foo.py]" in result
    assert "1 alpha" in result
    assert "2 beta" in result
    assert "3 gamma" in result
    assert "[end of foo.py]" in result


def test_build_prompt_wraps_files_with_line_numbers(template: Path) -> None:
    files = (retrieval.RetrievedFile(Path("m.py"), "a = 1\nb = 2"),)

    result = prompt.build_prompt("bug", files, template)

    assert "[start of m.py]" in result
    assert "1 a = 1" in result
    assert "2 b = 2" in result
    assert "[end of m.py]" in result
