"""Orchestrate SWE-bench Lite inference for one instance."""

import argparse
import json
import os
from pathlib import Path

import agent
import dataset
import one_shot


def main() -> None:
    """Load one SWE-bench instance, run inference, and save the patch.

    Raises:
        RuntimeError: When ANTHROPIC_API_KEY is missing or patch is empty.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    args = _build_parser().parse_args()
    output = args.output or Path(f"outputs/predictions_{args.mode}.jsonl")
    instance = dataset.load_instance(args.instance_id)
    patch = _run_patch(instance, args)
    if not patch:
        raise RuntimeError("Empty patch generated")

    save_prediction(instance.instance_id, _model_label(args.model), patch, output)
    print(f"Saved patch ({len(patch)} chars) to {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-bench inference.")
    parser.add_argument("--instance-id", default=dataset.DEFAULT_INSTANCE_ID)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--mode", choices=["agent", "one_shot"], default="agent")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path)
    return parser


def _run_patch(instance: dataset.SWEInstance, args: argparse.Namespace) -> str:
    repo_path = dataset.setup_repo(instance, args.workspace)
    match args.mode:
        case "agent":
            return agent.run_agent(repo_path, instance.problem_statement, args.model)
        case "one_shot":
            return one_shot.run_one_shot(repo_path, instance.problem_statement, args.model, args.top_k)
        case _:
            raise RuntimeError(f"Unknown mode: {args.mode}")


def _model_label(model_name: str) -> str:
    """Strip provider prefix from a model name.

    Args:
        model_name (str): Model name, optionally with provider prefix (e.g. anthropic/claude-sonnet-4-6).

    Returns:
        str: Model name without provider prefix (e.g. claude-sonnet-4-6).
    """
    return model_name.rsplit("/", 1)[-1]


def save_prediction(instance_id: str, model_label: str, patch: str, output_path: Path) -> None:
    """Append one prediction to a JSONL file in SWE-bench format.

    Args:
        instance_id (str): SWE-bench instance ID.
        model_label (str): Model label for output.
        patch (str): Unified diff string.
        output_path (Path): Output JSONL path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(_format_record(instance_id, model_label, patch))


def _format_record(instance_id: str, model_label: str, patch: str) -> str:
    return (
        json.dumps(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_label,
                "model_patch": patch,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
