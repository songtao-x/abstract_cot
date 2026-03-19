from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .gsm.script.gsm_utils import extract_first_integer, load_problem_from_json_blob, score_gsm_reasoning
except ImportError:
    TRAIN_DIR = Path(__file__).resolve().parent
    GSM_SCRIPT_DIR = TRAIN_DIR / "gsm" / "script"
    for candidate in (TRAIN_DIR, GSM_SCRIPT_DIR):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    from gsm_utils import extract_first_integer, load_problem_from_json_blob, score_gsm_reasoning


def _log_reward_stage(stage: str, message: str) -> None:
    print(f"[reward:{stage}] {message}", flush=True)


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


def extract_text(text: str, span: TagSpan | None) -> str:
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


def _countdown_answer_reward(answer_text: str, target: int, numbers: list[int]) -> float:
    parsed = _evaluate_answer(answer_text)
    if parsed is None:
        return 0.0

    value, used_numbers = parsed
    if value != Fraction(target):
        return 0.0
    if len(used_numbers) > 1 and sorted(used_numbers) != sorted(numbers):
        return 0.0
    return 1.0


def evaluate_countdown_completion(completion: str, target: int, numbers: list[int]) -> dict[str, Any]:
    parsed = process_response(completion)
    answer_text = extract_text(completion, parsed["answer"])
    normalized_answer = _normalize_answer_text(answer_text)
    evaluated = _evaluate_answer(answer_text)

    predicted_value: int | float | None = None
    used_numbers: list[int] = []
    failure_type = "ok"

    if parsed["answer"] is None:
        failure_type = "missing_answer"
    elif not normalized_answer:
        failure_type = "empty_answer"
    elif evaluated is None:
        failure_type = "parse_fail"
    else:
        value, used_numbers = evaluated
        predicted_value = int(value) if value.denominator == 1 else float(value)
        if value != Fraction(target):
            failure_type = "wrong_value"
        elif len(used_numbers) > 1 and sorted(used_numbers) != sorted(numbers):
            failure_type = "wrong_number_usage"

    base_reward = _countdown_answer_reward(answer_text, target, numbers)
    return {
        "base_reward": base_reward,
        "final_correct": base_reward == 1.0,
        "failure_type": failure_type if base_reward == 0.0 else "ok",
        "has_abstract_tag": parsed["abstract"] is not None,
        "has_answer_tag": parsed["answer"] is not None,
        "has_valid_tags": bool(parsed["can_score"]),
        "answer_text": answer_text,
        "normalized_answer": normalized_answer,
        "predicted_value": predicted_value,
        "used_numbers": used_numbers,
        "parsed": parsed,
    }


def evaluate_gsm_completion(completion: str, problem_json: str, target: int | None = None) -> dict[str, Any]:
    parsed = process_response(completion)
    think_text = extract_text(completion, parsed["think"])
    answer_text = extract_text(completion, parsed["answer"])
    predicted_answer = extract_first_integer(answer_text)

    parser_correct = False
    irrelevant_correct = False
    final_correct = False
    failure_type = "parse_fail"

    try:
        problem = load_problem_from_json_blob(problem_json)
        parser_correct, irrelevant_correct = score_gsm_reasoning(think_text, problem)
        gold_answer = int(problem.ans)
    except Exception:
        problem = None
        gold_answer = int(target) if target is not None else None

    if gold_answer is not None and predicted_answer is not None:
        final_correct = predicted_answer == gold_answer

    if parser_correct:
        failure_type = "ok"
    elif final_correct:
        failure_type = "wrong_reasoning"
    elif predicted_answer is None:
        failure_type = "parse_fail"
    else:
        failure_type = "wrong_final"

    base_reward = 1.0 if parser_correct else (0.25 if final_correct else 0.0)
    return {
        "base_reward": base_reward,
        "parser_correct": parser_correct,
        "irrelevant_correct": irrelevant_correct,
        "final_correct": final_correct,
        "predicted_answer": predicted_answer,
        "gold_answer": gold_answer,
        "failure_type": failure_type,
        "parsed": parsed,
        "think_text": think_text,
        "answer_text": answer_text,
    }


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _shift_span(span: TagSpan, offset: int) -> tuple[int, int]:
    return offset + span.content_start, offset + span.content_end


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
    for text, (char_start, char_end) in zip(full_texts, char_spans, strict=True):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:]

        if char_end <= char_start:
            span_logprobs.append(0.0)
            del encoded, input_ids, attention_mask, logits
            continue

        token_positions = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if end > start and end > char_start and start < char_end and idx > 0
        ]
        if not token_positions:
            span_logprobs.append(0.0)
            del encoded, input_ids, attention_mask, logits
            continue

        pred_positions = torch.tensor([idx - 1 for idx in token_positions], device=device)
        valid_mask = target_mask[0, pred_positions].bool()
        if not valid_mask.any():
            span_logprobs.append(0.0)
            del encoded, input_ids, attention_mask, logits
            continue

        pred_positions = pred_positions[valid_mask]
        token_ids = target_ids[0, pred_positions]
        log_probs = F.log_softmax(logits, dim=-1)
        span_logprobs.append(float(log_probs[0, pred_positions, token_ids].sum().item()))
        del encoded, input_ids, attention_mask, logits, log_probs

    return span_logprobs


def _apply_plan_delta(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    base_rewards: list[float],
    beta: float,
) -> list[float]:
    rewards: list[float] = []
    valid_examples: list[bool] = []
    full_with_plan_texts: list[str] = []
    full_without_plan_texts: list[str] = []
    think_with_spans: list[tuple[int, int]] = []
    think_without_spans: list[tuple[int, int]] = []
    answer_with_spans: list[tuple[int, int]] = []
    answer_without_spans: list[tuple[int, int]] = []

    _log_reward_stage("plan", f"building plan delta for batch_size={len(completions)}")

    for prompt, completion in zip(prompts, completions, strict=True):
        text = completion if isinstance(completion, str) else ""
        parsed = process_response(text)
        stripped = parsed["without_plan"]
        stripped_parsed = process_response(stripped)
        prompt_offset = len(prompt)

        can_score = (
            parsed["can_score"]
            and parsed["think"] is not None
            and parsed["answer"] is not None
            and stripped_parsed["think"] is not None
            and stripped_parsed["answer"] is not None
        )
        valid_examples.append(can_score)

        full_with_plan_texts.append(prompt + text)
        full_without_plan_texts.append(prompt + stripped)

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

    _log_reward_stage("plan", f"scorable examples={sum(valid_examples)}/{len(valid_examples)}")
    _log_reward_stage("plan", "scoring think-with spans")
    think_with_scores = _span_logprob_batch(model, tokenizer, full_with_plan_texts, think_with_spans)
    _log_reward_stage("plan", "scoring think-without spans")
    think_without_scores = _span_logprob_batch(model, tokenizer, full_without_plan_texts, think_without_spans)
    _log_reward_stage("plan", "scoring answer-with spans")
    answer_with_scores = _span_logprob_batch(model, tokenizer, full_with_plan_texts, answer_with_spans)
    _log_reward_stage("plan", "scoring answer-without spans")
    answer_without_scores = _span_logprob_batch(model, tokenizer, full_without_plan_texts, answer_without_spans)

    for idx, base_reward in enumerate(base_rewards):
        if not valid_examples[idx]:
            rewards.append(base_reward)
            continue

        delta = (think_with_scores[idx] - think_without_scores[idx]) - beta * (
            answer_with_scores[idx] - answer_without_scores[idx]
        )
        rewards.append(base_reward + delta)
    _log_reward_stage("plan", "plan delta computation done")
    return rewards


class CountdownPlanAwareReward:
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

        was_training = self.model.training
        self.model.eval()
        try:
            _log_reward_stage("countdown", f"computing base rewards for batch_size={len(completions)}")
            base_rewards = []
            for completion, gold, nums in zip(completions, target, numbers, strict=True):
                text = completion if isinstance(completion, str) else ""
                parsed = process_response(text)
                answer_text = extract_text(text, parsed["answer"])
                base_rewards.append(_countdown_answer_reward(answer_text, int(gold), [int(x) for x in nums]))
            return _apply_plan_delta(self.model, self.tokenizer, prompts, completions, base_rewards, self.beta)
        finally:
            if was_training:
                self.model.train()


class CountdownPureReward:
    def __init__(self) -> None:
        self.__name__ = self.__class__.__name__

    def bind_model(self, model: Any) -> None:
        del model

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        target: list[int],
        numbers: list[list[int]],
        **_: Any,
    ) -> list[float]:
        del prompts
        _log_reward_stage("countdown", f"computing pure rewards for batch_size={len(completions)}")
        rewards: list[float] = []
        for completion, gold, nums in zip(completions, target, numbers, strict=True):
            text = completion if isinstance(completion, str) else ""
            parsed = process_response(text)
            answer_text = extract_text(text, parsed["answer"])
            rewards.append(_countdown_answer_reward(answer_text, int(gold), [int(x) for x in nums]))
        return rewards


class GSMPlanAwareReward:
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
        problem_json: list[str],
        **_: Any,
    ) -> list[float]:
        if self.model is None:
            raise RuntimeError("Reward model is not bound.")

        was_training = self.model.training
        self.model.eval()
        try:
            _log_reward_stage("gsm", f"computing base rewards for batch_size={len(completions)}")
            base_rewards = []
            for completion, gold, problem_blob in zip(completions, target, problem_json, strict=True):
                text = completion if isinstance(completion, str) else ""
                metrics = evaluate_gsm_completion(text, problem_blob, target=int(gold))
                base_rewards.append(float(metrics["base_reward"]))
            return _apply_plan_delta(self.model, self.tokenizer, prompts, completions, base_rewards, self.beta)
        finally:
            if was_training:
                self.model.train()


class GSMPureReward:
    def __init__(self) -> None:
        self.__name__ = self.__class__.__name__

    def bind_model(self, model: Any) -> None:
        del model

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        target: list[int],
        problem_json: list[str],
        **_: Any,
    ) -> list[float]:
        del prompts
        _log_reward_stage("gsm", f"computing pure rewards for batch_size={len(completions)}")
        rewards: list[float] = []
        for completion, gold, problem_blob in zip(completions, target, problem_json, strict=True):
            text = completion if isinstance(completion, str) else ""
            metrics = evaluate_gsm_completion(text, problem_blob, target=int(gold))
            rewards.append(float(metrics["base_reward"]))
        return rewards


def build_reward(task: str, tokenizer: Any, beta: float, reward_variant: str = "plan") -> Any:
    if reward_variant not in {"plan", "pure"}:
        raise ValueError(f"Unsupported reward variant: {reward_variant}")

    if reward_variant == "pure":
        if task == "gsm":
            return GSMPureReward()
        if task == "countdown":
            return CountdownPureReward()
        raise ValueError(f"Unsupported task: {task}")

    if task == "gsm":
        return GSMPlanAwareReward(tokenizer=tokenizer, beta=beta)
    if task == "countdown":
        return CountdownPlanAwareReward(tokenizer=tokenizer, beta=beta)
    raise ValueError(f"Unsupported task: {task}")
