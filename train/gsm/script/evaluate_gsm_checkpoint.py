from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .gsm_prompt_template import gsm_abstract_prompt
    from .task_data import GSM_TASK_DESCRIPTION
    from .task_rewards import evaluate_gsm_completion
except ImportError:
    from gsm_prompt_template import gsm_abstract_prompt
    from task_data import GSM_TASK_DESCRIPTION
    from task_rewards import evaluate_gsm_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM evaluation for a saved checkpoint.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to the evaluation JSONL file (typically abstract/data/gsm_sample_test.jsonl).",
    )
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def load_rows(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_prompt(problem_text: str) -> str:
    return gsm_abstract_prompt.format(
        TASK_DESCIPTION=GSM_TASK_DESCRIPTION,
        PROBLEM_TEXT=problem_text.strip(),
    )


def infer_completion(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def classify_failure(metrics: dict[str, Any]) -> str:
    return str(metrics["failure_type"])


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    eval_file = Path(args.eval_file)
    output_file = Path(args.output_file) if args.output_file else model_dir / "gsm_eval.json"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    rows = load_rows(eval_file, max_samples=args.max_samples)
    results: list[dict[str, Any]] = []
    parser_correct = 0
    irrelevant_correct = 0
    final_correct = 0

    for row in rows:
        prompt = build_prompt(row["problem_text"])
        completion = infer_completion(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        metrics = evaluate_gsm_completion(completion, row["problem_json"], target=int(row["final_answer"]))
        parser_correct += int(metrics["parser_correct"])
        irrelevant_correct += int(metrics["irrelevant_correct"])
        final_correct += int(metrics["final_correct"])
        results.append(
            {
                "condition": row.get("condition", ""),
                "op": row.get("op"),
                "prompt": prompt,
                "completion": completion,
                "parser_correct": bool(metrics["parser_correct"]),
                "irrelevant_correct": bool(metrics["irrelevant_correct"]),
                "final_correct": bool(metrics["final_correct"]),
                "predicted_answer": metrics["predicted_answer"],
                "gold_answer": metrics["gold_answer"],
                "failure_type": classify_failure(metrics),
            }
        )

    total = len(results)
    summary = {
        "total": total,
        "parser_correct_rate": (parser_correct / total) if total else 0.0,
        "irrelevant_correct_rate": (irrelevant_correct / total) if total else 0.0,
        "final_answer_accuracy": (final_correct / total) if total else 0.0,
    }
    payload = {
        "model_dir": str(model_dir),
        "eval_file": str(eval_file),
        "summary": summary,
        "results": results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    print(f"Wrote GSM evaluation to {output_file}")


if __name__ == "__main__":
    main()
