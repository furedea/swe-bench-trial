"""Run mini-swe-agent on a repository and collect the resulting patch."""

import subprocess
from pathlib import Path

import model


def run_agent(repo_path: Path, problem_statement: str, model_name: str) -> str:
    """Run mini-swe-agent on a repository and return the resulting patch.

    Args:
        repo_path (Path): Path to the cloned repository.
        problem_statement (str): Issue description to solve.
        model_name (str): Model name (e.g. claude-sonnet-4-6 or anthropic/claude-sonnet-4-6).

    Returns:
        str: Unified diff string of changes made by the agent.

    Raises:
        subprocess.CalledProcessError: When mini-swe-agent exits with non-zero status.
    """
    subprocess.run(_mini_cmd(problem_statement, model_name), cwd=repo_path, check=True)
    return collect_patch(repo_path)


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
