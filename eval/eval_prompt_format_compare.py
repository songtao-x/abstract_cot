"""Quick A/B eval: greedy accuracy under the two prompt formats.

Compares the *training* prompt format (`enable_thinking` defaults to True, so
no auto-injected `<think></think>` block) against the *eval* prompt format
(`enable_thinking=False`, which auto-injects `<think>\\n\\n</think>\\n\\n`
after the assistant marker). Reports accuracy on the same problems for both,
plus a few example completions from each side for inspection.

Usage:
    python eval/eval_prompt_format_compare.py \\
        --checkpoint_dir /path/to/grpo_abstract_contrastive \\
        --eval_file /path/to/cd4_eval.jsonl \\
        --max_samples 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAIN_DIR = EVAL_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from prompt_template import abstract_prompt
from task_data import TASK_DESCRIPTION, load_countdown_dataset
from task_rewards import evaluate_countdown_completion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Path to checkpoint dir (the model dir, not the run root).")
    p.add_argument("--eval_file", type=str,
                   default=str(EVAL_DIR.parent / "data" / "cd4_eval.jsonl"))
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_examples_to_print", type=int, default=3)
    return p.parse_args()


def build_user_content(numbers: list[int], target: int) -> str:
    del target  # target is hidden from the model and only used by the reward function
    problem_text = (
        f"Numbers: {', '.join(str(v) for v in numbers)}\n"
        "Find a valid expression that reaches the target using each listed number exactly once."
    )
    return abstract_prompt.format(
        TASK_DESCIPTION=TASK_DESCRIPTION,
        PROBLEM_TEXT=problem_text,
    ) + "/no_think"


def build_prompt(tokenizer, numbers, target, *, enable_thinking_kw):
    """Build a chat-template prompt under one of the two formats.

    enable_thinking_kw=None  -> mimics training (kwarg not passed; Qwen3 default = True)
    enable_thinking_kw=False -> mimics current eval (auto-injects <think></think>)
    """
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": build_user_content(numbers, target)},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking_kw is not None:
        kwargs["enable_thinking"] = enable_thinking_kw
    return tokenizer.apply_chat_template(messages, **kwargs)


def run_format(model, tokenizer, dataset, *, enable_thinking_kw, max_new_tokens, label):
    import torch
    device = next(model.parameters()).device
    correct = 0
    results = []
    for row in dataset:
        prompt = build_prompt(
            tokenizer, row["numbers"], row["target"],
            enable_thinking_kw=enable_thinking_kw,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True,
        ).strip()
        ev = evaluate_countdown_completion(completion, row["target"], row["numbers"])
        is_correct = bool(ev["final_correct"])
        correct += int(is_correct)
        results.append({
            "numbers": row["numbers"],
            "target": row["target"],
            "completion": completion,
            "answer_text": ev["answer_text"],
            "failure_type": ev["failure_type"],
            "correct": is_correct,
        })
        if (len(results) % 20) == 0:
            print(f"  [{label}] {len(results)}/{len(dataset)}  running_acc={correct/len(results):.3f}",
                  flush=True)
    accuracy = correct / len(results) if results else 0.0
    return accuracy, results


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = Path(args.checkpoint_dir)
    print(f"checkpoint: {ckpt}")
    print(f"eval_file:  {args.eval_file}")
    print(f"max_samples: {args.max_samples}")
    print()

    dataset = load_countdown_dataset(Path(args.eval_file), max_samples=args.max_samples)
    print(f"loaded {len(dataset)} eval rows")

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # Show the two prompts on a single example so the difference is visible.
    sample = dataset[0]
    p_train = build_prompt(tokenizer, sample["numbers"], sample["target"], enable_thinking_kw=None)
    p_eval  = build_prompt(tokenizer, sample["numbers"], sample["target"], enable_thinking_kw=False)
    print("\n=== training-format prompt (last 120 chars) ===")
    print(repr(p_train[-120:]))
    print("=== eval-format prompt     (last 120 chars) ===")
    print(repr(p_eval[-120:]))
    print()

    print("running TRAINING format (enable_thinking kwarg NOT passed)...")
    acc_train, res_train = run_format(
        model, tokenizer, dataset,
        enable_thinking_kw=None,
        max_new_tokens=args.max_new_tokens,
        label="train",
    )

    print("\nrunning EVAL format (enable_thinking=False)...")
    acc_eval, res_eval = run_format(
        model, tokenizer, dataset,
        enable_thinking_kw=False,
        max_new_tokens=args.max_new_tokens,
        label="eval ",
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  training format (enable_thinking default=True):  acc = {acc_train:.3f}  ({sum(r['correct'] for r in res_train)}/{len(res_train)})")
    print(f"  eval     format (enable_thinking=False)        :  acc = {acc_eval:.3f}  ({sum(r['correct'] for r in res_eval)}/{len(res_eval)})")
    print(f"  gap (eval - train)                              :  {acc_eval - acc_train:+.3f}")
    print()

    # Failure-type breakdown so you can see WHY each format fails when it does.
    from collections import Counter
    print("training-format failure types:", dict(Counter(r["failure_type"] for r in res_train)))
    print("eval-format     failure types:", dict(Counter(r["failure_type"] for r in res_eval)))
    print()

    n = args.num_examples_to_print
    print(f"--- first {n} completions, training format ---")
    for r in res_train[:n]:
        print(f"  numbers={r['numbers']} target={r['target']} correct={r['correct']}")
        print(f"  completion: {r['completion'][:300]}")
        print()
    print(f"--- first {n} completions, eval format ---")
    for r in res_eval[:n]:
        print(f"  numbers={r['numbers']} target={r['target']} correct={r['correct']}")
        print(f"  completion: {r['completion'][:300]}")
        print()

    out_path = ckpt / "prompt_format_compare.json"
    with out_path.open("w") as f:
        json.dump({
            "checkpoint": str(ckpt),
            "eval_file": args.eval_file,
            "n_samples": len(dataset),
            "training_format_accuracy": acc_train,
            "eval_format_accuracy": acc_eval,
            "training_format_results": res_train,
            "eval_format_results": res_eval,
        }, f, indent=2)
    print(f"saved per-sample results -> {out_path}")


if __name__ == "__main__":
    main()
