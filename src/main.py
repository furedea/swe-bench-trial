"""Orchestrate SWE-bench Lite inference for one instance."""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import agent
import dataset
import one_shot


@dataclass(frozen=True, slots=True)
class PatchResult:
    """Result of patch generation for one SWE-bench instance."""

    instance_id: str
    model_label: str
    patch: str


def main() -> None:
    """Load one SWE-bench instance, run inference, and save the patch.

    Raises:
        RuntimeError: When ANTHROPIC_API_KEY is missing or patch is empty.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    args = _build_parser().parse_args()
    if args.output is None:
        args.output = Path(f"outputs/predictions_{args.mode}.jsonl")
    instance = dataset.load_instance(args.instance_id)
    patch = _run_patch(instance, args)
    if not patch:
        raise RuntimeError("Empty patch generated")

    result = PatchResult(instance.instance_id, _model_label(args.model), patch)
    save_prediction(result, args.output)
    print(f"Saved patch ({len(patch)} chars) to {args.output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-bench inference.")
    parser.add_argument("--instance-id", default=dataset.DEFAULT_INSTANCE_ID)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--mode", choices=["agent", "one_shot"], default="agent")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path(__file__).parent.parent / "prompts" / "prompt_template.txt",
    )
    return parser


def _run_patch(instance: dataset.SWEInstance, args: argparse.Namespace) -> str:
    dataset.setup_repo(instance, args.workspace)
    repo_path = args.workspace / instance.instance_id
    task = dataset.SWETask(repo_path, instance.problem_statement, args.model)
    match args.mode:
        case "agent":
            agent.run_agent(task)
            return agent.collect_patch(task.repo_path)
        case "one_shot":
            return one_shot.run_one_shot(task, args.top_k, args.prompt_template)
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


def save_prediction(result: PatchResult, output_path: Path) -> None:
    """Append one prediction to a JSONL file in SWE-bench format.

    Args:
        result (PatchResult): Patch generation result.
        output_path (Path): Output JSONL path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(_format_record(result))


def _format_record(result: PatchResult) -> str:
    return (
        json.dumps(
            {
                "instance_id": result.instance_id,
                "model_name_or_path": result.model_label,
                "model_patch": result.patch,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
