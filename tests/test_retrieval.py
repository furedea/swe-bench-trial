"""Tests for BM25-based file retrieval."""

from pathlib import Path

import retrieval


def _write_py(path: Path, content: str) -> None:
    path.write_text(content)


def test_retrieve_files_returns_top_k(tmp_path: Path) -> None:
    for i in range(5):
        _write_py(tmp_path / f"file{i}.py", f"def func{i}(): pass")

    results = retrieval.retrieve_files(tmp_path, "func0", top_k=3)

    assert len(results) == 3


def test_retrieve_files_ranks_relevant_file_higher(tmp_path: Path) -> None:
    _write_py(tmp_path / "units.py", "class Unit: conversion factor length meter")
    _write_py(tmp_path / "unrelated.py", "def hello(): return 'world'")

    results = retrieval.retrieve_files(tmp_path, "unit conversion factor", top_k=2)

    assert results[0].path.name == "units.py"


def test_retrieve_files_returns_file_content(tmp_path: Path) -> None:
    _write_py(tmp_path / "main.py", "x = 1\n")

    results = retrieval.retrieve_files(tmp_path, "main", top_k=1)

    assert results[0].content == "x = 1\n"


def test_retrieve_files_returns_empty_when_no_py_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# readme")

    results = retrieval.retrieve_files(tmp_path, "anything", top_k=5)

    assert results == ()


def test_retrieve_files_top_k_capped_by_file_count(tmp_path: Path) -> None:
    _write_py(tmp_path / "only.py", "x = 1")

    results = retrieval.retrieve_files(tmp_path, "query", top_k=10)

    assert len(results) == 1
