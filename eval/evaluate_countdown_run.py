from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
TRAIN_DIR = EVAL_DIR.parent / "train"

train_dir_str = str(TRAIN_DIR)
if train_dir_str not in sys.path:
    sys.path.insert(0, train_dir_str)

from prompt_template import abstract_prompt
from task_data import TASK_DESCRIPTION, load_countdown_dataset
from task_rewards import evaluate_countdown_completion


DEFAULT_RUN_DIR = TRAIN_DIR / "outputs" / "grpo_abstract"


def format_eval_log(stage: str, message: str, checkpoint: str | None = None) -> str:
    prefix = f"[countdown_eval:{stage}]"
    if checkpoint is None:
        return f"{prefix} {message}"
    return f"{prefix} checkpoint={checkpoint} {message}"


def build_checkpoint_postfix(checkpoint_name: str, completed: int, total: int) -> dict[str, str]:
    return {
        "ckpt": checkpoint_name,
        "done": f"{completed}/{total}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate countdown GRPO checkpoints and summarize the run.")
    parser.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--summary_file", type=str, default=None)
    return parser.parse_args()


def discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        suffix = child.name.split("-", 1)[1]
        if suffix.isdigit():
            checkpoints.append((int(suffix), child))
    return [path for _, path in sorted(checkpoints, key=lambda item: item[0])]


def _build_checkpoint_prompt(numbers: list[int], target: int) -> str:
    problem_text = (
        f"Numbers: {', '.join(str(value) for value in numbers)}\n"
        f"Target: {target}\n"
        "Find a valid expression that reaches the target using each listed number exactly once."
    )
    return abstract_prompt.format(
        TASK_DESCIPTION=TASK_DESCRIPTION,
        PROBLEM_TEXT=problem_text,
    )


def infer_completion(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def summarize_samples(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    accuracy = sum(int(item["final_correct"]) for item in results) / total if total else 0.0
    valid_tag_rate = sum(int(item["has_valid_tags"]) for item in results) / total if total else 0.0
    answer_tag_rate = sum(int(item["has_answer_tag"]) for item in results) / total if total else 0.0
    abstract_tag_rate = sum(int(item["has_abstract_tag"]) for item in results) / total if total else 0.0
    failure_counts = Counter(item["failure_type"] for item in results)
    return {
        "total": total,
        "accuracy": accuracy,
        "valid_tag_rate": valid_tag_rate,
        "answer_tag_rate": answer_tag_rate,
        "abstract_tag_rate": abstract_tag_rate,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def build_run_summary(checkpoint_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    ranking = sorted(
        (
            {
                "checkpoint": payload["checkpoint"],
                "accuracy": payload["summary"]["accuracy"],
                "total": payload["summary"]["total"],
            }
            for payload in checkpoint_payloads
        ),
        key=lambda item: (-item["accuracy"], item["checkpoint"]),
    )
    return {
        "num_checkpoints": len(checkpoint_payloads),
        "best_checkpoint": ranking[0]["checkpoint"] if ranking else None,
        "ranking": ranking,
    }


def evaluate_checkpoint(
    checkpoint_dir: Path,
    eval_file: Path,
    max_samples: int | None = None,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(
        format_eval_log(
            "dataset_load",
            f"eval_file={eval_file} max_samples={max_samples}",
            checkpoint=checkpoint_dir.name,
        ),
        flush=True,
    )
    effective_max_samples = 200 if max_samples is None else min(max_samples, 200)
    dataset = load_countdown_dataset(eval_file, max_samples=effective_max_samples)
    print(
        format_eval_log(
            "dataset_ready",
            f"samples={len(dataset)}",
            checkpoint=checkpoint_dir.name,
        ),
        flush=True,
    )
    print(format_eval_log("tokenizer_load", "loading tokenizer", checkpoint=checkpoint_dir.name), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(format_eval_log("model_load", "loading model", checkpoint=checkpoint_dir.name), flush=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype="auto")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    print(
        format_eval_log(
            "evaluate_start",
            f"running {len(dataset)} samples max_new_tokens={max_new_tokens}",
            checkpoint=checkpoint_dir.name,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for row in dataset:
        if not isinstance(row, dict):
            raise TypeError(
                "Expected countdown dataset rows as dict records; avoid slicing a Dataset with [:N] because "
                "it returns a dict-of-columns."
            )

        prompt = row.get("prompt") or _build_checkpoint_prompt(row["numbers"], int(row["target"]))
        completion = infer_completion(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        metrics = evaluate_countdown_completion(
            completion,
            target=int(row["target"]),
            numbers=[int(value) for value in row["numbers"]],
        )
        results.append(
            {
                "prompt": prompt,
                "completion": completion,
                "target": int(row["target"]),
                "numbers": [int(value) for value in row["numbers"]],
                "base_reward": float(metrics["base_reward"]),
                "final_correct": bool(metrics["final_correct"]),
                "failure_type": metrics["failure_type"],
                "has_abstract_tag": bool(metrics["has_abstract_tag"]),
                "has_answer_tag": bool(metrics["has_answer_tag"]),
                "has_valid_tags": bool(metrics["has_valid_tags"]),
                "answer_text": metrics["answer_text"],
                "predicted_value": metrics["predicted_value"],
                "used_numbers": metrics["used_numbers"],
            }
        )

    summary = summarize_samples(results)
    return {
        "checkpoint": checkpoint_dir.name,
        "checkpoint_dir": str(checkpoint_dir),
        "eval_file": str(eval_file),
        "summary": summary,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    eval_file = Path(args.eval_file) if args.eval_file else (TRAIN_DIR.parent / "data" / "cd4_eval.jsonl")

    if args.checkpoint_dir:
        checkpoints = [Path(args.checkpoint_dir)]
        run_dir = checkpoints[0].parent
    else:
        run_dir = Path(args.run_dir)
        checkpoints = discover_checkpoints(run_dir)

    if not checkpoints:
        raise ValueError(f"No checkpoints found under {run_dir}")

    print(
        format_eval_log(
            "startup",
            f"run_dir={run_dir} eval_file={eval_file} checkpoints={len(checkpoints)}",
        ),
        flush=True,
    )

    payloads: list[dict[str, Any]] = []
    total_checkpoints = len(checkpoints)
    progress = tqdm(checkpoints, desc="Countdown eval", unit="ckpt", dynamic_ncols=True, leave=True)
    for index, checkpoint_dir in enumerate(progress, start=1):
        progress.set_postfix(build_checkpoint_postfix(checkpoint_dir.name, index - 1, total_checkpoints), refresh=True)
        print(
            format_eval_log(
                "checkpoint_start",
                f"evaluating checkpoint_dir={checkpoint_dir}",
                checkpoint=checkpoint_dir.name,
            ),
            flush=True,
        )
        payload = evaluate_checkpoint(
            checkpoint_dir,
            eval_file=eval_file,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )
        payloads.append(payload)
        output_file = checkpoint_dir / "countdown_eval.json"
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        progress.set_postfix(build_checkpoint_postfix(checkpoint_dir.name, index, total_checkpoints), refresh=True)
        print(
            format_eval_log(
                "checkpoint_done",
                f"wrote checkpoint results to {output_file}",
                checkpoint=checkpoint_dir.name,
            ),
            flush=True,
        )
    progress.close()

    summary_payload = build_run_summary(payloads)
    summary_payload["run_dir"] = str(run_dir)
    summary_payload["eval_file"] = str(eval_file)
    summary_path = Path(args.summary_file) if args.summary_file else run_dir / "countdown_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=True)

    print(format_eval_log("summary_done", f"wrote countdown run summary to {summary_path}"), flush=True)


if __name__ == "__main__":
    main()
