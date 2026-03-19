"""Orchestrate SWE-bench Lite inference for one instance."""

import json
import os
from pathlib import Path

from dataset import DEFAULT_INSTANCE_ID, load_instance, setup_repo
from runner import run_agent


MODEL = "claude-sonnet-4-6"
WORKSPACE_DIR = Path("workspace")
OUTPUT_PATH = Path("outputs/predictions.jsonl")


def save_prediction(
    instance_id: str, model: str, patch: str, output_path: Path
) -> None:
    """Append one prediction to a JSONL file in SWE-bench format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": model,
                    "model_patch": patch,
                }
            )
            + "\n"
        )


def main() -> None:
    """Load one SWE-bench instance, run agent, and save the patch."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    instance = load_instance(DEFAULT_INSTANCE_ID)
    repo_path = setup_repo(instance, WORKSPACE_DIR)
    patch = run_agent(repo_path, instance.problem_statement, MODEL)

    if not patch:
        raise RuntimeError("Empty patch generated")

    save_prediction(instance.instance_id, MODEL, patch, OUTPUT_PATH)
    print(f"Saved patch ({len(patch)} chars) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
