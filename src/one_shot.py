"""Run one-shot inference pipeline: retrieve files, build prompt, infer patch."""

from pathlib import Path

import litellm
from swebench.inference.make_datasets.utils import extract_diff

import model
import prompt
import retrieval


def run_one_shot(repo_path: Path, problem_statement: str, model_name: str, top_k: int) -> str:
    """Retrieve relevant files, build a prompt, and return the extracted diff.

    Args:
        repo_path (Path): Root of the cloned repository.
        problem_statement (str): Issue description to solve.
        model_name (str): Model name (e.g. claude-sonnet-4-6 or anthropic/claude-sonnet-4-6).
        top_k (int): Number of files to retrieve for context.

    Returns:
        str: Extracted unified diff string, empty if none found.
    """
    files = retrieval.retrieve_files(repo_path, problem_statement, top_k)
    prompt_text = prompt.build_prompt(problem_statement, files)
    return _infer(prompt_text, model_name)


def _infer(prompt_text: str, model_name: str) -> str:
    response = litellm.completion(
        model=model.normalize_model_name(model_name),
        messages=[{"role": "user", "content": prompt_text}],
    )
    # content is None when the model returns an empty response
    content = response.choices[0].message.content or ""
    # extract_diff returns the original string when no diff block is found; fall back to ""
    return extract_diff(content) or ""
