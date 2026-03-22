"""Run mini-swe-agent on a repository and collect the resulting patch."""

import subprocess
from pathlib import Path

import dataset
import model


def run_agent(task: dataset.SWETask) -> None:
    """Run mini-swe-agent on a repository.

    Args:
        task (dataset.SWETask): Resolved SWE-bench task with repo, problem, and model.

    Raises:
        subprocess.CalledProcessError: When mini-swe-agent exits with non-zero status.
    """
    subprocess.run(_mini_cmd(task.problem_statement, task.model_name), cwd=task.repo_path, check=True)


def _mini_cmd(problem_statement: str, model_name: str) -> list[str]:
    return [
        "mini",
        "--model",
        model.normalize_model_name(model_name),
        "--task",
        problem_statement,
        "--yolo",  # skip interactive confirmation prompts
        "--exit-immediately",  # exit after task completion without waiting for input
    ]


def collect_patch(repo_path: Path) -> str:
    """Return git diff HEAD as a unified diff string.

    Args:
        repo_path (Path): Path to the git repository.

    Returns:
        str: Unified diff string, empty if no changes.

    Raises:
        subprocess.CalledProcessError: When git diff fails.
    """
    result = subprocess.run(
        # --no-ext-diff ensures plain unified diff output regardless of git diff driver config
        ["git", "diff", "--no-ext-diff", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
