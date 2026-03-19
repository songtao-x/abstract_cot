from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from datasets import Dataset
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from .cuda_alloc_conf import sanitize_pytorch_cuda_alloc_conf_for_vllm
except ImportError:
    from cuda_alloc_conf import sanitize_pytorch_cuda_alloc_conf_for_vllm

try:
    from .grpo_vllm_args import build_grpo_vllm_kwargs
except ImportError:
    from grpo_vllm_args import build_grpo_vllm_kwargs

try:
    from .wandb_utils import setup_wandb
except ImportError:
    from wandb_utils import setup_wandb

try:
    from .prompt_template import abstract_prompt
except ImportError:
    import importlib.util

    template_path = Path(__file__).resolve().parent / "prompt_template.py"
    spec = importlib.util.spec_from_file_location("prompt_template", template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompt template from {template_path}")
    prompt_template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_template)
    abstract_prompt = prompt_template.abstract_prompt

TASK_DESCRIPTION = (
    "Countdown arithmetic task: use the provided numbers to reach the target with +, -, *, /. "
    "Use each provided number exactly once and provide a valid solution in the required tag format."
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TRAIN_FILE = DATA_DIR / "cd4_train.jsonl"
DEFAULT_EVAL_FILE = DATA_DIR / "cd4_eval.jsonl"


def _log_stage(stage: str, message: str) -> None:
    print(f"[grpo:{stage}] {message}", flush=True)


def _parse_countdown_input(raw_input: str) -> tuple[list[int], int]:
    values = [int(x.strip()) for x in raw_input.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError(f"Invalid countdown input: {raw_input}")
    return values[:-1], values[-1]


def _build_problem_text(numbers: list[int], target: int) -> str:
    numbers_text = ", ".join(str(x) for x in numbers)
    return (
        f"Numbers: {numbers_text}\\n"
        f"Target: {target}\\n"
        "Find a valid expression that reaches the target using each listed number exactly once."
    )


# def data_process(data_path: str | Path, max_samples: int | None = None) -> Dataset:
#     path = Path(data_path)
#     rows: list[dict[str, Any]] = []

#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             item = json.loads(line)
#             numbers, target = _parse_countdown_input(item["input"])
#             prompt = abstract_prompt.format(
#                 TASK_DESCIPTION=TASK_DESCRIPTION,
#                 PROBLEM_TEXT=_build_problem_text(numbers, target),
#             )
#             rows.append(
#                 {
#                     "prompt": prompt + '/no_think\n',
#                     "target": target,
#                     "numbers": numbers,
#                     "reference": item.get("output", ""),
#                 }
#             )

#             if max_samples is not None and len(rows) >= max_samples:
#                 break

#     if not rows:
#         raise ValueError(f"No valid samples found in {path}")

#     return Dataset.from_list(rows)
def data_process(
    data_path: str | Path,
    tokenizer,
    max_samples: int | None = None,
) -> Dataset:
    path = Path(data_path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            numbers, target = _parse_countdown_input(item["input"])
            user_prompt = abstract_prompt.format(
                TASK_DESCIPTION=TASK_DESCRIPTION,
                PROBLEM_TEXT=_build_problem_text(numbers, target),
            )

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": user_prompt + "/no_think"},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            rows.append(
                {
                    "prompt": prompt,
                    "target": target,
                    "numbers": numbers,
                    "reference": item.get("output", ""),
                }
            )

            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"No valid samples found in {path}")

    return Dataset.from_list(rows)

@dataclass
class TagSpan:
    block_start: int
    block_end: int
    content_start: int
    content_end: int


def _find_tag_span(text: str, tag: str) -> TagSpan | None:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    block_start = text.find(open_tag)
    if block_start == -1:
        return None

    content_start = block_start + len(open_tag)
    close_start = text.find(close_tag, content_start)
    if close_start == -1:
        return None

    return TagSpan(
        block_start=block_start,
        block_end=close_start + len(close_tag),
        content_start=content_start,
        content_end=close_start,
    )


def _find_last_tag_span(text: str, tag: str) -> TagSpan | None:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    search_from = 0
    last_span: TagSpan | None = None

    while True:
        block_start = text.find(open_tag, search_from)
        if block_start == -1:
            break

        content_start = block_start + len(open_tag)
        close_start = text.find(close_tag, content_start)
        if close_start == -1:
            break

        last_span = TagSpan(
            block_start=block_start,
            block_end=close_start + len(close_tag),
            content_start=content_start,
            content_end=close_start,
        )
        search_from = last_span.block_end

    return last_span


def _content_span(start: int, end: int) -> TagSpan | None:
    if end < start:
        return None
    return TagSpan(
        block_start=start,
        block_end=end,
        content_start=start,
        content_end=end,
    )


def process_response(completion: str) -> dict[str, Any]:
    abstract = _find_tag_span(completion, "abstract")
    answer = _find_last_tag_span(completion, "answer")

    think_start = abstract.block_end if abstract is not None else 0
    think_end = answer.block_start if answer is not None else -1
    think = _content_span(think_start, think_end)

    without_plan = completion
    if abstract is not None:
        without_plan = completion[: abstract.block_start] + completion[abstract.block_end :]

    has_order = (
        abstract is not None
        and answer is not None
        and abstract.block_end <= answer.block_start
    )

    return {
        "abstract": abstract,
        "think": think,
        "answer": answer,
        "without_plan": without_plan,
        "can_score": has_order,
    }


def _extract_text(text: str, span: TagSpan | None) -> str:
    if span is None:
        return ""
    return text[span.content_start : span.content_end].strip()


def _normalize_answer_text(answer_text: str) -> str:
    answer_text = answer_text.strip()
    if "=" in answer_text:
        answer_text = answer_text.split("=", 1)[0].strip()
    return answer_text


def _eval_ast(node: ast.AST) -> tuple[Fraction, list[int]] | None:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        value = Fraction(str(node.value))
        used = [int(node.value)] if float(node.value).is_integer() else []
        return value, used

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        child = _eval_ast(node.operand)
        if child is None:
            return None
        value, used = child
        return ((value if isinstance(node.op, ast.UAdd) else -value), used)

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if left is None or right is None:
            return None

        left_value, left_used = left
        right_value, right_used = right

        if isinstance(node.op, ast.Add):
            value = left_value + right_value
        elif isinstance(node.op, ast.Sub):
            value = left_value - right_value
        elif isinstance(node.op, ast.Mult):
            value = left_value * right_value
        else:
            if right_value == 0:
                return None
            value = left_value / right_value

        return value, left_used + right_used

    return None


def _evaluate_answer(answer_text: str) -> tuple[Fraction, list[int]] | None:
    normalized = _normalize_answer_text(answer_text)
    if not normalized:
        return None

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return None

    return _eval_ast(tree)


def _final_answer_reward(answer_text: str, target: int, numbers: list[int]) -> float:
    parsed = _evaluate_answer(answer_text)
    if parsed is None:
        return 0.0

    value, used_numbers = parsed
    if value != Fraction(target):
        return 0.0

    if len(used_numbers) > 1 and sorted(used_numbers) != sorted(numbers):
        return 0.0

    return 1.0


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _shift_span(span: TagSpan, offset: int) -> tuple[int, int]:
    return offset + span.content_start, offset + span.content_end


# @torch.no_grad()
# def _span_logprob_batch(
#     model: Any,
#     tokenizer: Any,
#     full_texts: list[str],
#     char_spans: list[tuple[int, int]],
# ) -> list[float]:
#     if not full_texts:
#         return []

#     encoded = tokenizer(
#         full_texts,
#         return_tensors="pt",
#         add_special_tokens=False,
#         padding=True,
#         return_offsets_mapping=True,
#     )
#     offsets_batch = encoded.pop("offset_mapping").tolist()
#     input_ids = encoded["input_ids"].to(_model_device(model))
#     attention_mask = encoded["attention_mask"].to(input_ids.device)

#     logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
#     log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
#     target_ids = input_ids[:, 1:]
#     target_mask = attention_mask[:, 1:]

#     span_logprobs: list[float] = []
#     for batch_idx, ((char_start, char_end), offsets) in enumerate(zip(char_spans, offsets_batch, strict=True)):
#         if char_end <= char_start:
#             span_logprobs.append(0.0)
#             continue

#         token_positions = [
#             idx
#             for idx, (start, end) in enumerate(offsets)
#             if end > start and end > char_start and start < char_end and idx > 0
#         ]
#         if not token_positions:
#             span_logprobs.append(0.0)
#             continue

#         pred_positions = torch.tensor([idx - 1 for idx in token_positions], device=log_probs.device)
#         valid_mask = target_mask[batch_idx, pred_positions].bool()
#         if not valid_mask.any():
#             span_logprobs.append(0.0)
#             continue

#         pred_positions = pred_positions[valid_mask]
#         token_ids = target_ids[batch_idx, pred_positions]
#         span_logprobs.append(float(log_probs[batch_idx, pred_positions, token_ids].sum().item()))

#     return span_logprobs


@torch.no_grad()
def _span_logprob_batch(
    model: Any,
    tokenizer: Any,
    full_texts: list[str],
    char_spans: list[tuple[int, int]],
) -> list[float]:
    if not full_texts:
        return []

    model.eval()
    device = _model_device(model)

    span_logprobs: list[float] = []
    _log_stage("reward", f"scoring span logprobs for batch_size={len(full_texts)}")

    for text, (char_start, char_end) in zip(full_texts, char_spans, strict=True):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,  # important: no batch padding
            return_offsets_mapping=True,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()  # [T, 2]
        input_ids = encoded["input_ids"].to(device)          # [1, T]
        attention_mask = encoded["attention_mask"].to(device) # [1, T]

        # forward in mixed precision to reduce memory
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits  # [1,T,V]

        logits = logits[:, :-1, :]              # [1,T-1,V]
        target_ids = input_ids[:, 1:]           # [1,T-1]
        target_mask = attention_mask[:, 1:]     # [1,T-1]

        # Keep the same number of model forwards on every rank. With ZeRO-3, skipping
        # a forward on one rank desynchronizes DeepSpeed parameter all-gathers and can
        # deadlock NCCL when another rank is still inside model(...)
        token_positions = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if end > start and end > char_start and start < char_end and idx > 0
        ]
        if not token_positions:
            span_logprobs.append(0.0)
            del encoded, input_ids, attention_mask, logits
            continue

        pred_positions = torch.tensor([idx - 1 for idx in token_positions], device=device)  # positions in logits[:, :-1]

        valid_mask = target_mask[0, pred_positions].bool()
        if not valid_mask.any():
            span_logprobs.append(0.0)
            del encoded, input_ids, attention_mask, logits
            continue

        pred_positions = pred_positions[valid_mask]
        token_ids = target_ids[0, pred_positions]  # [K]

        # gather logprob of the actual next token at those positions
        # (this avoids indexing a huge tensor later; still uses log_softmax though)
        log_probs = F.log_softmax(logits, dim=-1)  # [1,T-1,V]  (ok for 1-by-1)
        lp = log_probs[0, pred_positions, token_ids].sum().item()

        span_logprobs.append(float(lp))

        # optional: help fragmentation
        del encoded, input_ids, attention_mask, logits, log_probs
        # torch.cuda.empty_cache()  # usually not needed; use only if fragmentation is bad

    return span_logprobs

class PlanAwareReward:
    def __init__(self, tokenizer: Any, beta: float) -> None:
        self.tokenizer = tokenizer
        self.beta = beta
        self.model: Any | None = None
        self.__name__ = self.__class__.__name__

    def bind_model(self, model: Any) -> None:
        self.model = model

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        target: list[int],
        numbers: list[list[int]],
        **_: Any,
    ) -> list[float]:
        if self.model is None:
            raise RuntimeError("Reward model is not bound.")

        rewards: list[float] = []
        was_training = self.model.training
        self.model.eval()

        try:
            _log_stage("reward", f"computing countdown rewards for batch_size={len(completions)}")
            base_rewards: list[float] = []
            valid_examples: list[bool] = []
            full_with_plan_texts: list[str] = []
            full_without_plan_texts: list[str] = []
            think_with_spans: list[tuple[int, int]] = []
            think_without_spans: list[tuple[int, int]] = []
            answer_with_spans: list[tuple[int, int]] = []
            answer_without_spans: list[tuple[int, int]] = []

            for prompt, completion, gold, nums in zip(prompts, completions, target, numbers):
                text = completion if isinstance(completion, str) else ""
                parsed = process_response(text)
                answer_text = _extract_text(text, parsed["answer"])
                base_reward = _final_answer_reward(answer_text, int(gold), [int(x) for x in nums])
                base_rewards.append(base_reward)

                stripped = parsed["without_plan"]
                stripped_parsed = process_response(stripped)
                prompt_offset = len(prompt)
                full_with_plan = prompt + text
                full_without_plan = prompt + stripped

                can_score = (
                    parsed["can_score"]
                    and parsed["think"] is not None
                    and parsed["answer"] is not None
                    and stripped_parsed["think"] is not None
                    and stripped_parsed["answer"] is not None
                )
                valid_examples.append(can_score)

                full_with_plan_texts.append(full_with_plan)
                full_without_plan_texts.append(full_without_plan)

                if can_score:
                    think_with_spans.append(_shift_span(parsed["think"], prompt_offset))
                    think_without_spans.append(_shift_span(stripped_parsed["think"], prompt_offset))
                    answer_with_spans.append(_shift_span(parsed["answer"], prompt_offset))
                    answer_without_spans.append(_shift_span(stripped_parsed["answer"], prompt_offset))
                else:
                    zero_span = (0, 0)
                    think_with_spans.append(zero_span)
                    think_without_spans.append(zero_span)
                    answer_with_spans.append(zero_span)
                    answer_without_spans.append(zero_span)

            _log_stage("reward", f"plan-delta scorable examples={sum(valid_examples)}/{len(valid_examples)}")

            think_with_scores = _span_logprob_batch(
                self.model,
                self.tokenizer,
                full_with_plan_texts,
                think_with_spans,
            )
            _log_stage("reward", "think-with span scores done")
            think_without_scores = _span_logprob_batch(
                self.model,
                self.tokenizer,
                full_without_plan_texts,
                think_without_spans,
            )
            _log_stage("reward", "think-without span scores done")
            answer_with_scores = _span_logprob_batch(
                self.model,
                self.tokenizer,
                full_with_plan_texts,
                answer_with_spans,
            )
            _log_stage("reward", "answer-with span scores done")
            answer_without_scores = _span_logprob_batch(
                self.model,
                self.tokenizer,
                full_without_plan_texts,
                answer_without_spans,
            )

            for idx, base_reward in enumerate(base_rewards):
                if not valid_examples[idx]:
                    rewards.append(base_reward)
                    continue

                delta = (
                    think_with_scores[idx] - think_without_scores[idx]
                ) - self.beta * (
                    answer_with_scores[idx] - answer_without_scores[idx]
                )

                _log_stage("reward", f"delta reward={delta}")
                rewards.append(base_reward + delta)
        finally:
            if was_training:
                self.model.train()

        return rewards


def train(args: argparse.Namespace) -> None:
    _log_stage("start", f"model={args.model_name}")
    run_name = setup_wandb(
        task="countdown",
        loss_type="grpo",
        reward_variant="plan",
        model_name=args.model_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        log_fn=lambda message: _log_stage("wandb", message),
    )
    alloc_conf_before = sanitize_pytorch_cuda_alloc_conf_for_vllm(use_vllm=False)
    alloc_conf_after = sanitize_pytorch_cuda_alloc_conf_for_vllm(use_vllm=args.use_vllm)
    if alloc_conf_before != alloc_conf_after:
        _log_stage(
            "env",
            "removed expandable_segments:True from PYTORCH_CUDA_ALLOC_CONF for vLLM compatibility",
        )
    _log_stage("tokenizer", "loading tokenizer")
    # if "Qwen" in args.model_name:
    #     enable_thinking = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log_stage("model", "loading model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")

    _log_stage("data", f"loading train dataset from {args.train_file}")
    train_ds = data_process(args.train_file, tokenizer=tokenizer, max_samples=args.max_train_samples)
    _log_stage("data", f"train dataset size={len(train_ds)}")

    _log_stage("deepspeed", f"loading config from {args.ds_cfg}")
    with open(args.ds_cfg, 'r') as f:
        ds_cfg = json.load(f)

        ds_cfg["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    

    eval_ds = None
    eval_strategy = "no"
    if not args.no_eval and args.eval_file and Path(args.eval_file).exists():
        _log_stage("data", f"loading eval dataset from {args.eval_file}")
        eval_ds = data_process(args.eval_file, tokenizer=tokenizer, max_samples=args.max_eval_samples)
        eval_strategy = "steps"
        _log_stage("data", f"eval dataset size={len(eval_ds)}")
    else:
        _log_stage("data", "evaluation disabled or eval file missing")
    
    _log_stage("config", "building GRPOConfig")
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        generation_batch_size=args.generation_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name=run_name,
        **build_grpo_vllm_kwargs(args),
        # deepspeed
        deepspeed=ds_cfg,
    )

    _log_stage("reward", "building countdown reward function")
    reward_fn = PlanAwareReward(tokenizer=tokenizer, beta=args.reward_beta)
    _log_stage("trainer", "initializing GRPOTrainer")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    reward_fn.bind_model(trainer.model)

    _log_stage("train", "starting trainer.train()")
    trainer.train()
    _log_stage("save", f"saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _log_stage("done", "training pipeline finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRPO on countdown data with abstract prompt format.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--eval_file", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "grpo_abstract"))

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=256)
    parser.add_argument("--no_eval", action="store_true")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--num_generations_eval", type=int, default=1)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--generation_batch_size", type=int, default=4)

    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=15)
    parser.add_argument("--eval_steps", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward_beta", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate")
    parser.add_argument("--vllm_model_impl", type=str, default="vllm")
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true")
    parser.add_argument("--vllm_server_base_url", type=str, default=None)
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_server_timeout", type=float, default=240.0)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--vllm_max_model_length", type=int, default=4096)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    parser.add_argument("--ds_cfg", default="/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/ds_zero3.json")
    parser.add_argument("--wandb_project", type=str, default="abstract_grpo_runs")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Use either --bf16 or --fp16, not both.")
    train(args)


if __name__ == "__main__":
    main()
