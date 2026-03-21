"""SWE-bench Lite instance loading and repository setup."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import datasets


DEFAULT_INSTANCE_ID = "astropy__astropy-12907"


@dataclass(frozen=True, slots=True)
class SWEInstance:
    """A single SWE-bench task instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Self:
        """Construct from a HuggingFace dataset row.

        Args:
            row (dict[str, Any]): Dataset row with instance_id, repo, base_commit, problem_statement.

        Returns:
            Self: Constructed SWEInstance.

        Raises:
            KeyError: When a required field is missing from row.
        """
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
        )


def load_instance(instance_id: str = DEFAULT_INSTANCE_ID) -> SWEInstance:
    """Load one instance from SWE-bench Lite test split.

    Args:
        instance_id (str): SWE-bench instance ID.

    Returns:
        SWEInstance: The matching instance.

    Raises:
        ValueError: When the instance ID is not found in the dataset.
    """
    dataset = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    rows = [r for r in dataset if r["instance_id"] == instance_id]
    if not rows:
        raise ValueError(f"Instance not found: {instance_id}")
    return SWEInstance.from_dict(rows[0])


def setup_repo(instance: SWEInstance, workspace_dir: Path) -> Path:
    """Clone the instance repo at base_commit into workspace_dir.

    Skips cloning if the directory already exists.

    Args:
        instance (SWEInstance): SWE-bench instance with repo and base_commit.
        workspace_dir (Path): Directory to clone into.

    Returns:
        Path: Path to the cloned repository root.
    """
    repo_path = workspace_dir / instance.instance_id
    if repo_path.exists():
        return repo_path
    _clone(instance, repo_path, workspace_dir)
    return repo_path


def _clone(instance: SWEInstance, repo_path: Path, workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", f"https://github.com/{instance.repo}.git", str(repo_path)],
        check=True,
    )
    subprocess.run(["git", "checkout", instance.base_commit], cwd=repo_path, check=True)
