"""Build one-shot prompts from a problem statement and retrieved files."""

from pathlib import Path

import retrieval


def build_prompt(
    problem_statement: str,
    files: tuple[retrieval.RetrievedFile, ...],
    prompt_template: Path,
) -> str:
    """Build a prompt following prompt_style_2 from Jimenez et al. 2023 (SWE-bench).

    Args:
        problem_statement (str): Issue description to solve.
        files (tuple[retrieval.RetrievedFile, ...]): Retrieved files, highest relevance first.
        prompt_template (Path): Path to the prompt template file with {problem_statement} and {code} placeholders.

    Returns:
        str: Formatted prompt with line-numbered file context and a few-shot patch example.
    """
    return prompt_template.read_text().format(
        problem_statement=problem_statement,
        code=_format_files(files),
    )


def _format_files(files: tuple[retrieval.RetrievedFile, ...]) -> str:
    # Line numbers let the model reference specific lines in the generated hunk headers.
    entries = (_format_file_with_lines(f.path, f.content) for f in files)
    return "\n".join(entries)


def _format_file_with_lines(path: str, content: str) -> str:
    lines = content.splitlines()
    numbered = "\n".join(f"{i + 1} {line}" for i, line in enumerate(lines))
    return f"[start of {path}]\n{numbered}\n[end of {path}]"
