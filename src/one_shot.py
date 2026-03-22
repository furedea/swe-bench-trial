"""Run one-shot inference pipeline: retrieve files, build prompt, infer patch."""

import re
from pathlib import Path

import litellm

import dataset
import model
import prompt
import retrieval


def run_one_shot(task: dataset.SWETask, top_k: int, prompt_template: Path) -> str:
    """Retrieve relevant files, build a prompt, and return the extracted patch.

    Args:
        task (dataset.SWETask): Resolved SWE-bench task with repo, problem, and model.
        top_k (int): Number of files to retrieve for context.
        prompt_template (Path): Path to the prompt template file.

    Returns:
        str: Unified diff extracted from model output; empty string if no diff found.
    """
    files = retrieval.retrieve_files(task.repo_path, task.problem_statement, top_k)
    prompt_text = prompt.build_prompt(task.problem_statement, files, prompt_template)
    content = _call_model(prompt_text, task.model_name)
    return extract_diff(content)


def _call_model(prompt_text: str, model_name: str) -> str:
    response = litellm.completion(
        model=model.normalize_model_name(model_name),
        messages=[{"role": "user", "content": prompt_text}],
    )
    return response.choices[0].message.content or ""


def extract_diff(content: str) -> str:
    """Extract a unified diff from raw model output.

    Args:
        content (str): Raw text returned by the model.

    Returns:
        str: Unified diff string; empty string if no diff pattern is found.
    """
    # Primary: model follows the few-shot format and wraps diff in <patch> tags.
    match = re.search(r"<patch>(.*?)</patch>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback 1: model uses a fenced code block instead.
    match = re.search(r"```(?:diff)?\s*\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback 2: raw diff with no wrapper.
    match = re.search(r"(--- a/.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
