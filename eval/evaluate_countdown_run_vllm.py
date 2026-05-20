"""vLLM-based countdown evaluator for GRPO checkpoints.

The HF `model.generate()` path in evaluate_countdown_run.py runs one prompt at
a time and is the bottleneck on multi-checkpoint sweeps. This version loads
each checkpoint into a vLLM engine, batch-generates the entire eval set in a
single call, and tears the engine down before moving on.

Prompt format matches the training pipeline post-no-think removal: no
`/no_think` suffix, default chat template (Qwen3 thinking mode enabled).

Usage:
  python evaluate_countdown_run_vllm.py \
      --run_dir <outputs/grpo_abstract_xxx> \
      --eval_file <abstract/data/cd4_eval.jsonl> \
      [--max_samples 200] \
      [--max_new_tokens 2048] \
      [--every_n_checkpoints 1] \
      [--gpu_memory_utilization 0.9]
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
TRAIN_DIR = EVAL_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from prompt_template import abstract_prompt
from task_data import TASK_DESCRIPTION, load_countdown_dataset
from task_rewards import evaluate_countdown_completion


DEFAULT_RUN_DIR = TRAIN_DIR / "outputs" / "grpo_abstract"


def _log(stage: str, message: str, ckpt: str | None = None) -> None:
    head = f"[cd_eval_vllm:{stage}]"
    if ckpt is not None:
        head = f"{head} ckpt={ckpt}"
    print(f"{head} {message}", flush=True)


def discover_checkpoints(run_dir: Path) -> list[Path]:
    pairs: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        suffix = child.name.split("-", 1)[1]
        if suffix.isdigit():
            pairs.append((int(suffix), child))
    return [p for _, p in sorted(pairs, key=lambda x: x[0])]


def _build_chat_prompt(tokenizer: Any, numbers: list[int], target: int) -> str:
    """Build prompt matching training format (grpo._build_problem_text).

    The training prompt includes ``Target: {target}`` — the model needs the
    target to solve countdown. The legacy ``task_data._build_countdown_problem_text``
    and ``evaluate_countdown_run._build_chat_prompt`` both stripped the target
    via ``del target`` (with a misleading comment about it being hidden), which
    is why prior evals capped at near-zero accuracy.
    """
    problem_text = (
        f"Numbers: {', '.join(str(v) for v in numbers)}\n"
        f"Target: {target}\n"
        "Find a valid expression that reaches the target using each listed number exactly once."
    )
    user_content = abstract_prompt.format(
        TASK_DESCIPTION=TASK_DESCRIPTION,
        PROBLEM_TEXT=problem_text,
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0, "accuracy": 0.0}
    failure_counts = Counter(item["failure_type"] for item in results)
    return {
        "total": total,
        "accuracy": sum(int(r["final_correct"]) for r in results) / total,
        "valid_tag_rate": sum(int(r["has_valid_tags"]) for r in results) / total,
        "answer_tag_rate": sum(int(r["has_answer_tag"]) for r in results) / total,
        "abstract_tag_rate": sum(int(r["has_abstract_tag"]) for r in results) / total,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def evaluate_checkpoint_vllm(
    checkpoint_dir: Path,
    eval_file: Path,
    *,
    max_samples: int | None,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
    dtype: str,
) -> dict[str, Any]:
    # Imports kept local so the top-level script can be imported without vLLM.
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    _log("dataset_load", f"eval_file={eval_file} max_samples={max_samples}", ckpt=checkpoint_dir.name)
    effective_max = 200 if max_samples is None else min(max_samples, 200)
    dataset = load_countdown_dataset(eval_file, max_samples=effective_max)
    _log("dataset_ready", f"samples={len(dataset)}", ckpt=checkpoint_dir.name)

    _log("tokenizer_load", "loading tokenizer", ckpt=checkpoint_dir.name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict[str, Any]] = []
    prompts: list[str] = []
    for row in dataset:
        if not isinstance(row, dict):
            raise TypeError("Expected dataset rows as dicts; do not slice a HF Dataset with [:N].")
        numbers = [int(v) for v in row["numbers"]]
        target = int(row["target"])
        prompts.append(_build_chat_prompt(tokenizer, numbers, target))
        rows.append({"numbers": numbers, "target": target})

    _log("model_load", "loading vLLM engine", ckpt=checkpoint_dir.name)
    t_load = time.time()
    llm = LLM(
        model=str(checkpoint_dir),
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=False,
        trust_remote_code=True,
    )
    _log("model_ready", f"loaded in {time.time() - t_load:.1f}s", ckpt=checkpoint_dir.name)

    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    _log("generate_start", f"batch_size={len(prompts)} max_new_tokens={max_new_tokens}", ckpt=checkpoint_dir.name)
    t_gen = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
    gen_secs = time.time() - t_gen
    _log("generate_done", f"{gen_secs:.1f}s ({gen_secs / max(len(prompts), 1):.2f}s/sample)", ckpt=checkpoint_dir.name)

    results: list[dict[str, Any]] = []
    for row, prompt, out in zip(rows, prompts, outputs):
        completion = out.outputs[0].text
        metrics = evaluate_countdown_completion(
            completion,
            target=row["target"],
            numbers=row["numbers"],
        )
        results.append({
            "prompt": prompt,
            "completion": completion,
            "target": row["target"],
            "numbers": row["numbers"],
            "base_reward": float(metrics["base_reward"]),
            "final_correct": bool(metrics["final_correct"]),
            "failure_type": metrics["failure_type"],
            "has_abstract_tag": bool(metrics["has_abstract_tag"]),
            "has_answer_tag": bool(metrics["has_answer_tag"]),
            "has_valid_tags": bool(metrics["has_valid_tags"]),
            "answer_text": metrics["answer_text"],
            "predicted_value": metrics["predicted_value"],
            "used_numbers": metrics["used_numbers"],
        })

    summary = summarize(results)
    _log("summary",
         f"acc={summary['accuracy']:.4f}  valid_tag={summary.get('valid_tag_rate', 0.0):.3f}",
         ckpt=checkpoint_dir.name)

    # Tear down the engine before moving on. vLLM doesn't expose a clean
    # shutdown — drop refs and let the GC collect, then reset CUDA.
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_dir.name,
        "checkpoint_dir": str(checkpoint_dir),
        "eval_file": str(eval_file),
        "summary": summary,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM-batched countdown eval over GRPO checkpoints.")
    p.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_DIR))
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="If set, eval only this single checkpoint (overrides --run_dir discovery).")
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=200,
                   help="Cap eval samples per checkpoint (default 200).")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--summary_file", type=str, default=None)
    p.add_argument("--every_n_checkpoints", type=int, default=1,
                   help="Sub-sample checkpoints (e.g. 2 evaluates every other checkpoint).")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip checkpoints that already have countdown_eval.json written.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_file = Path(args.eval_file) if args.eval_file else (TRAIN_DIR.parent / "data" / "cd4_eval.jsonl")

    if args.checkpoint_dir:
        checkpoints = [Path(args.checkpoint_dir)]
        run_dir = checkpoints[0].parent
    else:
        run_dir = Path(args.run_dir)
        checkpoints = discover_checkpoints(run_dir)
        if args.every_n_checkpoints > 1:
            checkpoints = checkpoints[:: args.every_n_checkpoints] + (
                [checkpoints[-1]] if checkpoints and (len(checkpoints) - 1) % args.every_n_checkpoints != 0 else []
            )

    if not checkpoints:
        raise ValueError(f"No checkpoints found under {run_dir}")

    _log("startup",
         f"run_dir={run_dir} eval_file={eval_file} checkpoints={len(checkpoints)} "
         f"max_samples={args.max_samples} max_new_tokens={args.max_new_tokens}")

    payloads: list[dict[str, Any]] = []
    for index, checkpoint_dir in enumerate(checkpoints, start=1):
        out_path = checkpoint_dir / "countdown_eval.json"
        if args.skip_existing and out_path.exists():
            _log("skip", f"{out_path} exists — skipping", ckpt=checkpoint_dir.name)
            with out_path.open("r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
            continue

        _log("checkpoint_start", f"[{index}/{len(checkpoints)}] {checkpoint_dir}", ckpt=checkpoint_dir.name)
        payload = evaluate_checkpoint_vllm(
            checkpoint_dir,
            eval_file=eval_file,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
        )
        payloads.append(payload)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        _log("checkpoint_done", f"wrote {out_path}", ckpt=checkpoint_dir.name)

    ranking = sorted(
        ({"checkpoint": p["checkpoint"], "accuracy": p["summary"]["accuracy"], "total": p["summary"]["total"]}
         for p in payloads),
        key=lambda r: (-r["accuracy"], r["checkpoint"]),
    )
    summary_payload = {
        "run_dir": str(run_dir),
        "eval_file": str(eval_file),
        "num_checkpoints": len(payloads),
        "best_checkpoint": ranking[0]["checkpoint"] if ranking else None,
        "ranking": ranking,
    }
    summary_path = Path(args.summary_file) if args.summary_file else run_dir / "countdown_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=True)
    _log("summary_done", f"wrote {summary_path}")


if __name__ == "__main__":
    main()
