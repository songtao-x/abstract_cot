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

from task_data import load_gsm_dataset
from task_rewards import evaluate_gsm_completion


DEFAULT_RUN_DIR = TRAIN_DIR / "outputs" / "grpo_gsm"


def format_eval_log(stage: str, message: str, checkpoint: str | None = None) -> str:
    prefix = f"[gsm_eval:{stage}]"
    if checkpoint is None:
        return f"{prefix} {message}"
    return f"{prefix} checkpoint={checkpoint} {message}"


def build_checkpoint_postfix(checkpoint_name: str, completed: int, total: int) -> dict[str, str]:
    return {
        "ckpt": checkpoint_name,
        "done": f"{completed}/{total}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSM GRPO checkpoints and summarize the run.")
    parser.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=200)
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


def infer_completion(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def summarize_samples(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    final_answer_accuracy = sum(int(item["final_correct"]) for item in results) / total if total else 0.0
    parser_correct_rate = sum(int(item["parser_correct"]) for item in results) / total if total else 0.0
    irrelevant_correct_rate = sum(int(item["irrelevant_correct"]) for item in results) / total if total else 0.0
    avg_base_reward = sum(float(item["base_reward"]) for item in results) / total if total else 0.0
    failure_counts = Counter(str(item["failure_type"]) for item in results)
    return {
        "total": total,
        "final_answer_accuracy": final_answer_accuracy,
        "parser_correct_rate": parser_correct_rate,
        "irrelevant_correct_rate": irrelevant_correct_rate,
        "avg_base_reward": avg_base_reward,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def build_run_summary(checkpoint_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    ranking = sorted(
        (
            {
                "checkpoint": payload["checkpoint"],
                "final_answer_accuracy": payload["summary"].get("final_answer_accuracy", 0.0),
                "parser_correct_rate": payload["summary"].get("parser_correct_rate", 0.0),
                "irrelevant_correct_rate": payload["summary"].get("irrelevant_correct_rate", 0.0),
                "avg_base_reward": payload["summary"].get("avg_base_reward", 0.0),
                "total": payload["summary"].get("total", 0),
            }
            for payload in checkpoint_payloads
        ),
        key=lambda item: (-item["final_answer_accuracy"], item["checkpoint"]),
    )
    return {
        "num_checkpoints": len(checkpoint_payloads),
        "best_checkpoint": ranking[0]["checkpoint"] if ranking else None,
        "ranking": ranking,
    }


def evaluate_checkpoint(
    checkpoint_dir: Path,
    eval_file: Path,
    max_samples: int = 200,
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
    effective_max_samples = min(max_samples, 200)
    dataset = load_gsm_dataset(eval_file, max_samples=effective_max_samples)
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
                "Expected GSM dataset rows as dict records; avoid slicing a Dataset with [:N] because "
                "it returns a dict-of-columns."
            )

        prompt = str(row.get("prompt", ""))
        if not prompt:
            raise ValueError("Missing prompt in GSM eval row; load_gsm_dataset should provide prompt.")
        completion = infer_completion(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        metrics = evaluate_gsm_completion(
            completion,
            str(row["problem_json"]),
            target=int(row["target"]),
        )
        results.append(
            {
                "prompt": prompt,
                "completion": completion,
                "target": int(row["target"]),
                "reference": row.get("reference", ""),
                "base_reward": float(metrics["base_reward"]),
                "parser_correct": bool(metrics["parser_correct"]),
                "irrelevant_correct": bool(metrics["irrelevant_correct"]),
                "final_correct": bool(metrics["final_correct"]),
                "predicted_answer": metrics["predicted_answer"],
                "gold_answer": metrics["gold_answer"],
                "failure_type": metrics["failure_type"],
                "think_text": metrics["think_text"],
                "answer_text": metrics["answer_text"],
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
    eval_file = Path(args.eval_file) if args.eval_file else (TRAIN_DIR.parent / "data" / "gsm_sample_test.jsonl")

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
    progress = tqdm(checkpoints, desc="GSM eval", unit="ckpt", dynamic_ncols=True, leave=True)
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
        output_file = checkpoint_dir / "gsm_eval.json"
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
    summary_path = Path(args.summary_file) if args.summary_file else run_dir / "gsm_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=True)

    print(format_eval_log("summary_done", f"wrote gsm run summary to {summary_path}"), flush=True)


if __name__ == "__main__":
    main()
