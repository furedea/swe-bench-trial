"""SWE-bench Lite instance loading and repository setup."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from datasets import load_dataset


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
        """Construct from a HuggingFace dataset row. KeyError propagates naturally."""
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
        )


def load_instance(instance_id: str = DEFAULT_INSTANCE_ID) -> SWEInstance:
    """Load one instance from SWE-bench Lite test split."""
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    rows = [r for r in dataset if r["instance_id"] == instance_id]
    if not rows:
        raise ValueError(f"Instance not found: {instance_id}")
    return SWEInstance.from_dict(rows[0])


def setup_repo(instance: SWEInstance, workspace_dir: Path) -> Path:
    """Clone repo at base_commit into workspace_dir. Skip if already exists."""
    repo_path = workspace_dir / instance.instance_id
    if repo_path.exists():
        return repo_path

    workspace_dir.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{instance.repo}.git"
    subprocess.run(["git", "clone", clone_url, str(repo_path)], check=True)
    subprocess.run(["git", "checkout", instance.base_commit], cwd=repo_path, check=True)
    return repo_path
