"""Run mini-swe-agent on a repository and collect the resulting patch."""

import subprocess
from pathlib import Path


def run_agent(repo_path: Path, problem_statement: str, model: str) -> str:
    """Run mini-swe-agent and return the git diff of changes made."""
    subprocess.run(
        [
            "mini",
            "--model",
            f"anthropic/{model}",
            "--task",
            problem_statement,
            "--yolo",
            "--exit-immediately",
        ],
        cwd=repo_path,
        check=True,
    )
    return collect_patch(repo_path)


def collect_patch(repo_path: Path) -> str:
    """Return git diff HEAD as a unified diff string."""
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
