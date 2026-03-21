"""BM25-based file retrieval from a repository."""

from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass(frozen=True, slots=True)
class RetrievedFile:
    path: Path
    content: str


def retrieve_files(repo_path: Path, query: str, top_k: int = 10) -> tuple[RetrievedFile, ...]:
    """Retrieve the most relevant Python files from a repo using BM25.

    Args:
        repo_path (Path): Root of the cloned repository.
        query (str): Issue description used as the search query.
        top_k (int): Maximum number of files to return.

    Returns:
        tuple[RetrievedFile, ...]: Ranked list of RetrievedFile, highest relevance first.
    """
    # sorted for deterministic ordering; rglob does not guarantee traversal order
    py_files = sorted(repo_path.rglob("*.py"))
    if not py_files:
        return ()
    return _rank(py_files, query, top_k)


def _rank(py_files: list[Path], query: str, top_k: int) -> tuple[RetrievedFile, ...]:
    # errors="replace" to handle binary or non-UTF-8 files without crashing
    contents = [p.read_text(errors="replace") for p in py_files]
    bm25 = BM25Okapi([doc.split() for doc in contents])
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(scores, py_files, contents), key=lambda x: x[0], reverse=True)
    return tuple(RetrievedFile(path, content) for _, path, content in ranked[:top_k])
