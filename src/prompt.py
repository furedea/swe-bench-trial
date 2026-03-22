"""Build one-shot prompts from a problem statement and retrieved files."""

import retrieval


def build_prompt(problem_statement: str, files: tuple[retrieval.RetrievedFile, ...]) -> str:
    """Build a one-shot prompt with the problem statement and retrieved file contents.

    Args:
        problem_statement (str): Issue description to solve.
        files (tuple[retrieval.RetrievedFile, ...]): Retrieved files, highest relevance first.

    Returns:
        str: Formatted prompt ready for one-shot LM inference.
    """
    sections = [
        problem_statement,
        *_format_files(files),
        "Please output a unified diff patch to resolve the issue.",
    ]
    return "\n\n".join(sections)


def _format_files(files: tuple[retrieval.RetrievedFile, ...]) -> tuple[str, ...]:
    if not files:
        return ()
    entries = tuple(f"{f.path}\n```\n{f.content}\n```" for f in files)
    return ("The following files may be relevant to fixing the issue.", *entries)
