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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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
        f"Numbers: {numbers_text}\n"
        f"Target: {target}\n"
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
                {"role": "user", "content": user_prompt},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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


def _parse_expression(expr_text: str) -> tuple[Fraction, list[int]] | None:
    """Parse a single arithmetic expression (no '=' splitting)."""
    expr_text = expr_text.strip()
    if not expr_text:
        return None
    try:
        tree = ast.parse(expr_text, mode="eval")
    except SyntaxError:
        return None
    return _eval_ast(tree)


def _validate_multistep_answer(
    answer_text: str, target: int, numbers: list[int]
) -> bool:
    """Validate a comma-separated multi-step countdown solution.

    Format example: ``90+95=185, 11*185=2035, 2035/37=55``

    Rules enforced:
      1. Each step is ``<expr> = <value>`` and the LHS evaluates to the RHS.
      2. Atoms appearing in each LHS must either be one of the given numbers
         (consumed at most once across all steps) or an intermediate value
         produced by an earlier step's RHS.
      3. Every given number is consumed exactly once across the full solution.
      4. The final step's RHS equals ``target``.

    Returns True only if all rules pass.
    """
    text = answer_text.strip()
    if not text or "=" not in text:
        return False

    raw_steps = [s for s in (s.strip() for s in text.split(",")) if s]
    if not raw_steps:
        return False

    remaining = list(numbers)        # multiset of unconsumed given numbers
    intermediates: set[int] = set()  # values produced by earlier steps
    last_value: Fraction | None = None

    for step in raw_steps:
        if "=" not in step:
            return False
        lhs_text, rhs_text = step.split("=", 1)

        lhs = _parse_expression(lhs_text)
        if lhs is None:
            return False
        lhs_value, lhs_atoms = lhs

        try:
            rhs_value = Fraction(rhs_text.strip())
        except (ValueError, ZeroDivisionError):
            return False

        if lhs_value != rhs_value:
            return False

        # Each atom must come from the given-numbers pool or from a prior step.
        # Consume from `remaining` first so reused intermediates don't double-spend.
        for atom in lhs_atoms:
            if atom in remaining:
                remaining.remove(atom)
            elif atom in intermediates:
                continue
            else:
                return False  # atom not in given numbers and not an intermediate

        if rhs_value.denominator == 1:
            intermediates.add(int(rhs_value))
        last_value = rhs_value

    if last_value != Fraction(target):
        return False
    if remaining:
        return False  # at least one given number was never used
    return True


def _final_answer_reward(answer_text: str, target: int, numbers: list[int]) -> float:
    """Reward 1.0 iff the answer is a valid countdown solution.

    Two accepted answer formats:
      * Multi-step: comma-separated ``expr=value`` chain whose final value
        equals the target. Each given number must be consumed exactly once
        across the chain; intermediate atoms must come from earlier RHS values.
      * Single expression: one arithmetic expression that uses every given
        number exactly once and evaluates to the target. e.g. ``(90+95)*11/37``.

    Both cases require: (a) the expression(s) only use the given numbers
    (each exactly once) and (b) the result equals the target.
    """
    if not isinstance(answer_text, str):
        return 0.0
    text = answer_text.strip()
    if not text:
        return 0.0

    # Multi-step path: any answer containing '=' is treated as a step chain.
    if "=" in text:
        if _validate_multistep_answer(text, target, numbers):
            return 1.0
        # If multi-step validation failed but the text looks like a single
        # ``expr = value`` (no commas), fall through to single-expression
        # evaluation below — _normalize_answer_text will strip the '=...' tail.

    parsed = _evaluate_answer(text)
    if parsed is None:
        return 0.0

    value, used_numbers = parsed
    # Validation: must reach the target AND use exactly the given numbers
    # (multiset equality — each given number used once, no extras).
    if value != Fraction(target):
        return 0.0
    if sorted(used_numbers) != sorted(numbers):
        return 0.0
    return 1.0


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
) -> list[tuple[float, float, int]]:
    """Return list of (logprob_sum, entropy_sum, token_count) per example.

    Per-token averages: logprob_sum / token_count and entropy_sum / token_count.
    Entropy at each position is the full predictive-distribution entropy
    H = -sum_v p_v * log p_v (in nats). Sum is over the span's predicted positions.
    When no valid tokens are found, returns (0.0, 0.0, 0).
    """
    if not full_texts:
        return []

    model.eval()
    device = _model_device(model)

    span_results: list[tuple[float, float, int]] = []
    _log_stage("reward", f"scoring span logprobs+entropy for batch_size={len(full_texts)}")

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
            span_results.append((0.0, 0.0, 0))
            del encoded, input_ids, attention_mask, logits
            continue

        pred_positions = torch.tensor([idx - 1 for idx in token_positions], device=device)  # positions in logits[:, :-1]

        valid_mask = target_mask[0, pred_positions].bool()
        if not valid_mask.any():
            span_results.append((0.0, 0.0, 0))
            del encoded, input_ids, attention_mask, logits
            continue

        pred_positions = pred_positions[valid_mask]
        token_ids = target_ids[0, pred_positions]  # [K]

        # gather logprob of the actual next token at those positions
        # (this avoids indexing a huge tensor later; still uses log_softmax though)
        log_probs = F.log_softmax(logits, dim=-1)  # [1,T-1,V]  (ok for 1-by-1)
        lp = log_probs[0, pred_positions, token_ids].sum().item()
        # Token-level entropy of the predictive distribution at each scored position.
        # H_t = -sum_v exp(log p_v) * log p_v ; reported in nats.
        span_log_probs = log_probs[0, pred_positions, :]                      # [K, V]
        ent = -(span_log_probs.exp() * span_log_probs).sum(dim=-1).sum().item()
        n_tokens = pred_positions.shape[0]

        span_results.append((float(lp), float(ent), n_tokens))

        # optional: help fragmentation
        del encoded, input_ids, attention_mask, logits, log_probs, span_log_probs
        # torch.cuda.empty_cache()  # usually not needed; use only if fragmentation is bad

    return span_results


@torch.no_grad()
def _plan_attention_score(
    model: Any,
    tokenizer: Any,
    text: str,
    plan_char_span: tuple[int, int],
    answer_char_span: tuple[int, int],
) -> float | None:
    """Normalized plan attention ratio using the last quarter of transformer layers.

    For each answer token the softmax attention already sums to 1, so we measure
    what *fraction* of that budget lands on plan tokens, then divide by the naive
    uniform baseline (plan_len / total_len).  A ratio > 1 means the answer attends
    to the plan more than chance; < 1 means it mostly ignores it.

    Metric = mean over {last-quarter layers, all query heads, answer tokens} of:
        sum_{j in plan} attn[answer_i, j]  /  (plan_len / total_len)

    Aligned to Qwen3-4B (36 layers, 32 query heads / 8 KV heads GQA):
    - In eager / SDPA-with-output_attentions mode HF expands KV heads to Q heads,
      so each layer's attention tensor is [1, 32, T, T].
    - Last quarter = layers 27-35 (computed dynamically from actual layer count).
    - With flash_attention_2 the model raises ValueError; we catch it and return None.

    Returns None if attention weights are unavailable.
    """
    plan_start, plan_end = plan_char_span
    ans_start, ans_end = answer_char_span

    device = _model_device(model)
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
    total_len = input_ids.shape[1]

    # Truncate to bound attention memory while keeping both plan (start) and answer (end)
    # inside the window. Plan is anchored at the START. Answer sits at the END and can
    # spill past _MAX_ATTN_LEN when prompt + completion exceeds the cap; in that case
    # we expand the window just far enough to cover the answer span.
    #
    # The patched eager function uses SDPA (no T×T) + a partial [H, A, T] matrix in bfloat16.
    # At T=3000, A=50: [1, 32, 50, 3000] bfloat16 ≈ 9.6 MB — negligible. Even at T≈4500
    # (prompt + max_completion_length=2048 + margin) the per-layer footprint stays small.
    _MAX_ATTN_LEN = 3000
    has_valid_spans = plan_start < plan_end and ans_start < ans_end
    keep_len = min(total_len, _MAX_ATTN_LEN)
    if has_valid_spans and total_len > _MAX_ATTN_LEN:
        ans_token_end = 0
        for i, (s, e) in enumerate(offsets):
            if e > s and s < ans_end and e > ans_start:
                ans_token_end = i + 1
        if ans_token_end > _MAX_ATTN_LEN:
            keep_len = min(total_len, ans_token_end + 16)
    if total_len > keep_len:
        input_ids = input_ids[:, :keep_len]
        attention_mask = attention_mask[:, :keep_len]
        offsets = offsets[:keep_len]
        total_len = keep_len

    # Compute positions before the model call so we know whether to score afterward,
    # but do NOT return early here — we must always call model() to keep ZeRO-3
    # all-gather counts in sync across ranks.
    if has_valid_spans:
        plan_positions = torch.tensor(
            [i for i, (s, e) in enumerate(offsets) if e > s and e > plan_start and s < plan_end],
            device=device,
        )
        answer_positions = torch.tensor(
            [i for i, (s, e) in enumerate(offsets) if e > s and e > ans_start and s < ans_end],
            device=device,
        )
    else:
        plan_positions = torch.zeros(0, dtype=torch.long, device=device)
        answer_positions = torch.zeros(0, dtype=torch.long, device=device)

    can_score = plan_positions.numel() > 0 and answer_positions.numel() > 0

    # Unwrap DDP / DeepSpeed engine to reach the HF model and its layer list.
    _hf_model = model.module if hasattr(model, "module") else model
    _orig_impl = getattr(_hf_model.config, "_attn_implementation", "sdpa")

    _hf_layers = _hf_model.model.layers
    num_layers = len(_hf_layers)
    last_quarter_start = (num_layers * 3) // 4   # e.g. layer 27 for Qwen3-4B (36 layers)

    plan_len = plan_positions.numel() if can_score else 0
    baseline = plan_len / total_len if plan_len > 0 else 1.0

    # Monkey-patch eager_attention_forward for the duration of this forward pass.
    #
    # WHY: The stock eager_attention_forward allocates a full [1, H, T, T] float32 tensor
    # for softmax — 512 MB at T=2048, before any hook can intervene.
    # FIX: replace it with a version that uses F.scaled_dot_product_attention for the
    # attention OUTPUT (fused kernel, no T×T allocation) and only computes a tiny
    # [1, H, A, T] partial matrix (A = #answer tokens ≈ 50) for plan-mass scoring.
    #
    # Dispatch path in Qwen3Attention.forward:
    #   attention_interface: Callable = eager_attention_forward   # module-level name lookup
    #   if _attn_implementation != "eager":
    #       attention_interface = ALL_ATTENTION_FUNCTIONS[...]
    # Patching the module attribute before the forward call replaces the lookup target.
    import transformers.models.qwen3.modeling_qwen3 as _qwen3_mod
    _orig_eager = _qwen3_mod.eager_attention_forward
    layer_scores: list[float] = []

    def _efficient_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        # Expand KV heads for GQA (Qwen3-4B: 32 Q heads, 8 KV heads → repeat ×4).
        key_exp = _qwen3_mod.repeat_kv(key, module.num_key_value_groups)   # [1, H, T, D]
        val_exp = _qwen3_mod.repeat_kv(value, module.num_key_value_groups) # [1, H, T, D]

        # Causal mask: [1, 1, T, T] with 0/-inf values, or None.
        causal_mask = attention_mask[:, :, :, : key_exp.shape[-2]] if attention_mask is not None else None

        # Attention OUTPUT via SDPA — fused kernel, never materialises [H, T, T].
        attn_output = F.scaled_dot_product_attention(
            query, key_exp, val_exp, attn_mask=causal_mask, dropout_p=dropout, scale=scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()  # [1, T, H, D]

        # Plan-attention scoring — only for last-quarter layers when spans are valid.
        # [1, H, A, T] is tiny (≈6 MB at A=50, T=2048) vs [1, H, T, T] (≈512 MB float32).
        if can_score and hasattr(module, "layer_idx") and module.layer_idx >= last_quarter_start:
            q_ans = query[:, :, answer_positions, :]                          # [1, H, A, D]
            scores = torch.matmul(q_ans, key_exp.transpose(-2, -1)) * scaling # [1, H, A, T]
            if causal_mask is not None:
                scores = scores + causal_mask[:, :, answer_positions, :]
            attn = F.softmax(scores, dim=-1)  # stays bfloat16 — [1, H, A, T]
            plan_mass = attn[:, :, :, plan_positions].sum(dim=-1)             # [1, H, A]
            layer_scores.append((plan_mass / baseline).mean().item())
            del q_ans, scores, attn, plan_mass

        del key_exp, val_exp
        return attn_output, None  # None: no T×T tensor returned or accumulated

    # All ranks switch config together (each has its own copy) so ZeRO-3 all-gathers stay in sync.
    _hf_model.config._attn_implementation = "eager"
    _qwen3_mod.eager_attention_forward = _efficient_eager
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,  # no attention tensors accumulated
            )
    finally:
        _hf_model.config._attn_implementation = _orig_impl
        _qwen3_mod.eager_attention_forward = _orig_eager

    del encoded, input_ids, attention_mask, outputs

    if not can_score:
        return None

    if not layer_scores:
        raise RuntimeError(
            "_plan_attention_score: no layer scores collected despite can_score=True. "
            "Check that Qwen3Attention sets self.layer_idx and the last-quarter range is valid."
        )
    return sum(layer_scores) / len(layer_scores)


class PlanAwareReward:
    def __init__(
        self,
        tokenizer: Any,
        beta: float,
        output_dir: str | None = None,
        contrastive_cot: bool = False,
        contrastive_weight: float = 0.3,
        contrastive_max_tokens: int = 2048,
        think_delta_weight: float = 1.0,
        base_reward_weight: float = 1.0,
        think_delta_clip: float = 0.0,
        think_min_tokens: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.beta = beta
        self.think_delta_weight = think_delta_weight
        self.base_reward_weight = base_reward_weight
        self.think_delta_clip = think_delta_clip
        self.think_min_tokens = think_min_tokens
        self.contrastive_cot = contrastive_cot
        self.contrastive_weight = contrastive_weight
        self.contrastive_max_tokens = contrastive_max_tokens
        self.model: Any | None = None
        # Optional reference to trl.generation.VLLMGeneration; populated after the
        # trainer is constructed via bind_vllm_engine. When set, the contrastive
        # baseline is generated with vLLM (orders of magnitude faster than HF
        # generate under DeepSpeed ZeRO-3, which all-gathers params at every
        # token-step).
        self.vllm_engine: Any | None = None
        self.__name__ = self.__class__.__name__
        self._output_dir = output_dir
        self._call_count = 0
        self._metrics_file: Any | None = None
        import os
        _is_main = os.environ.get("LOCAL_RANK", "0") == "0"
        if output_dir is not None and _is_main:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self._metrics_file = open(Path(output_dir) / "reward_metrics.jsonl", "a", encoding="utf-8")

    def _reg_think_delta(self, raw_delta: float, tw_n: int) -> float:
        """Apply anti-Goodhart regularizers to the raw think logprob delta.

        The think-delta term is per-token-normalized, so the policy can inflate
        it *without improving the task* by collapsing `think` into a short,
        trivially plan-derivable stub — observed empirically in
        dapo_abstract_think_only_n16: think_delta climbs to ~0.30 while
        think_chars falls 156→100 and base_reward stays flat
        (corr(think_delta, base) ≈ −0.09).

        Two optional, independently-gated guards (both off when 0):
          * clip   — bound |delta| to ±think_delta_clip so the term cannot
                      dominate the O(1)-scale base reward, regardless of how
                      far the policy pushes per-token logprob.
          * length — linearly ramp the term to 0 below think_min_tokens
                      (smooth, not a hard cliff GRPO would itself game at the
                      boundary), removing the incentive to shorten `think`.
        """
        d = raw_delta
        if self.think_delta_clip > 0.0:
            c = self.think_delta_clip
            d = max(-c, min(c, d))
        if self.think_min_tokens > 0:
            d *= min(1.0, tw_n / self.think_min_tokens)
        return d

    def bind_model(self, model: Any) -> None:
        self.model = model
        self._verify_attention_or_die()

    def bind_vllm_engine(self, engine: Any) -> None:
        """Attach the trainer's vLLM generation wrapper for contrastive sampling.

        ``engine`` is a ``trl.generation.VLLMGeneration`` instance (the trainer
        exposes it as ``trainer.vllm_generation``). When bound, contrastive
        no-plan rollouts go through vLLM instead of ``model.generate()``.
        """
        self.vllm_engine = engine
        if engine is not None and self.contrastive_cot:
            _log_stage("contrastive", "vLLM engine bound — using vLLM for no-plan rollouts")

    @torch.no_grad()
    def _verify_attention_or_die(self) -> None:
        """Smoke-test that the efficient attention patch will work at runtime.

        Verifies two things before any training step:
          1. The model has transformer layers accessible at model.model.layers.
          2. Each attention module exposes a layer_idx attribute (relied on by the patch).
        Raises RuntimeError immediately so the job dies at startup rather than wasting GPU hours.
        """
        _hf_model = self.model.module if hasattr(self.model, "module") else self.model
        try:
            layers = _hf_model.model.layers
        except AttributeError:
            raise RuntimeError(
                "PlanAwareReward: cannot find _hf_model.model.layers. "
                "The attention score metric requires a Qwen3-style model with .model.layers."
            )
        missing = [i for i, l in enumerate(layers) if not hasattr(l.self_attn, "layer_idx")]
        if missing:
            raise RuntimeError(
                f"PlanAwareReward: self_attn.layer_idx missing on layers {missing[:5]}. "
                "The efficient attention patch relies on layer_idx to identify last-quarter layers."
            )
        import transformers.models.qwen3.modeling_qwen3 as _qwen3_mod
        if not hasattr(_qwen3_mod, "eager_attention_forward"):
            raise RuntimeError(
                "PlanAwareReward: transformers.models.qwen3.modeling_qwen3 has no "
                "eager_attention_forward — cannot install the memory-efficient patch."
            )
        _log_stage("reward", "attention smoke-test passed — efficient attention patch is ready")

    @torch.no_grad()
    @torch.no_grad()
    def _generate_contrastive_completions_vllm(
        self,
        prompts: list[str],
        num_rollouts: int = 1,
    ) -> list[list[str]]:
        """vLLM path for the contrastive baseline.

        Mirrors the colocate pattern used inside ``trl.generation.VLLMGeneration``:
        gather the per-rank prompt slices across the TP group, run a single
        ``llm.generate`` collectively, then slice each rank's share back out.
        Wakes / sleeps the KV cache when sleep mode is enabled, matching the
        trainer's lifecycle. Skipping uninformative cases is handled at the
        reward-add site so the gather stays symmetric across ranks.
        """
        from vllm import SamplingParams

        engine = self.vllm_engine
        no_plan_prefix = "<think>\n</think>\n"
        prefixes = [p + no_plan_prefix for p in prompts]

        _log_stage(
            "contrastive",
            f"vLLM: generating {num_rollouts} no-plan rollouts × {len(prompts)} prompts",
        )

        sampling = SamplingParams(
            n=num_rollouts,
            temperature=1.0,
            top_p=1.0,
            max_tokens=self.contrastive_max_tokens,
        )

        if getattr(engine, "enable_sleep_mode", False):
            engine.llm.wake_up(tags=["kv_cache"])

        tp_size = getattr(engine, "tensor_parallel_size", 1) or 1
        if tp_size > 1:
            gathered: list[Any] = [None for _ in range(tp_size)]
            torch.distributed.all_gather_object(gathered, prefixes, group=engine.tp_group)
            all_prefixes = [p for sub in gathered for p in sub]  # type: ignore[arg-type]
        else:
            all_prefixes = prefixes

        outputs = engine.llm.generate(all_prefixes, sampling_params=sampling, use_tqdm=False)

        if tp_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=engine.tp_group)
            n_local = len(prefixes)
            outputs = outputs[local_rank_in_group * n_local : (local_rank_in_group + 1) * n_local]

        if getattr(engine, "enable_sleep_mode", False):
            engine.llm.sleep(level=2)

        all_rollouts: list[list[str]] = []
        for out in outputs:
            rolls: list[str] = []
            for sub in out.outputs:
                rolls.append(no_plan_prefix + sub.text)
            all_rollouts.append(rolls)

        _log_stage(
            "contrastive",
            f"vLLM: generated {sum(len(r) for r in all_rollouts)} no-plan rollouts",
        )
        return all_rollouts

    def _generate_contrastive_completions(
        self,
        prompts: list[str],
        num_rollouts: int = 1,
    ) -> list[list[str]]:
        """Generate sampled no-plan rollouts as the contrastive baseline.

        Force-feeds a closed empty ``<think></think>`` block (skipping both the
        ``<abstract>`` block and any reasoning), so the baseline lands right at
        the ``<answer>`` boundary. Sampling ``num_rollouts`` continuations with
        non-zero temperature keeps the baseline from collapsing to a single
        greedy edge case.

        Batched: under DeepSpeed ZeRO-3 every forward pass triggers a parameter
        all-gather across ranks. Looping per-prompt pays that gather cost B
        times per token-step; one batched generate pays it once.
        """
        _log_stage(
            "contrastive",
            f"generating {num_rollouts} no-plan rollouts × {len(prompts)} prompts (batched)",
        )
        device = _model_device(self.model)
        _hf_model = self.model.module if hasattr(self.model, "module") else self.model

        no_plan_prefix = "<think>\n</think>\n"
        prefixes = [p + no_plan_prefix for p in prompts]
        batch_size = len(prefixes)

        # Left-pad so all prefixes end at the same column — required for HF
        # generate to advance every sequence's cursor in lock-step.
        prev_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        try:
            encoded = self.tokenizer(
                prefixes,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = prev_padding_side

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]  # uniform across the batch (left-padded)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = _hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.contrastive_max_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=num_rollouts,
                pad_token_id=pad_id,
            )

        # output_ids shape: [batch_size * num_rollouts, prompt_len + new_len].
        # HF lays out rollouts contiguously: rows [i*k : (i+1)*k] belong to prompt i.
        new_tokens = output_ids[:, prompt_len:]  # strip the (left-padded) prompt
        all_rollouts: list[list[str]] = []
        for batch_idx in range(batch_size):
            rollouts: list[str] = []
            for r in range(num_rollouts):
                seq_idx = batch_idx * num_rollouts + r
                generated = self.tokenizer.decode(
                    new_tokens[seq_idx], skip_special_tokens=False
                )
                rollouts.append(no_plan_prefix + generated)
            all_rollouts.append(rollouts)

        _log_stage(
            "contrastive",
            f"generated {sum(len(r) for r in all_rollouts)} no-plan rollouts",
        )
        return all_rollouts

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
            abstract_lens: list[float] = []
            think_lens: list[float] = []
            parsed_results: list[dict] = []

            for prompt, completion, gold, nums in zip(prompts, completions, target, numbers):
                text = completion if isinstance(completion, str) else ""
                parsed = process_response(text)
                parsed_results.append(parsed)
                answer_text = _extract_text(text, parsed["answer"])
                base_reward = _final_answer_reward(answer_text, int(gold), [int(x) for x in nums])
                base_rewards.append(base_reward)
                abstract_lens.append(float(
                    parsed["abstract"].content_end - parsed["abstract"].content_start
                    if parsed["abstract"] is not None else 0
                ))
                think_lens.append(float(
                    parsed["think"].content_end - parsed["think"].content_start
                    if parsed["think"] is not None else 0
                ))

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

            valid_indices = [i for i, v in enumerate(valid_examples) if v]
            n_total = len(valid_examples)

            # --- metrics always available every step ---
            metrics: dict[str, Any] = {
                "step":                            self._call_count,
                "reward/base_reward_mean":         sum(base_rewards) / n_total,
                "reward/valid_fraction":           len(valid_indices) / n_total,
                "lengths/abstract_chars_mean":     sum(abstract_lens) / n_total,
                "lengths/think_chars_mean":        sum(think_lens)    / n_total,
            }

            # --- logprob-delta + entropy metrics: only when scorable examples exist ---
            if valid_indices:
                # Unpack (logprob_sum, entropy_sum, token_count) tuples → per-token averages
                tw_avg  = [think_with_scores[i][0]    / think_with_scores[i][2]    if think_with_scores[i][2]    > 0 else 0.0 for i in valid_indices]
                tow_avg = [think_without_scores[i][0] / think_without_scores[i][2] if think_without_scores[i][2] > 0 else 0.0 for i in valid_indices]
                aw_avg  = [answer_with_scores[i][0]   / answer_with_scores[i][2]   if answer_with_scores[i][2]   > 0 else 0.0 for i in valid_indices]
                aow_avg = [answer_without_scores[i][0]/ answer_without_scores[i][2]if answer_without_scores[i][2]> 0 else 0.0 for i in valid_indices]
                # Per-token entropy on the with-plan rollouts (the policy's own generation)
                tw_ent  = [think_with_scores[i][1]    / think_with_scores[i][2]    if think_with_scores[i][2]    > 0 else 0.0 for i in valid_indices]
                aw_ent  = [answer_with_scores[i][1]   / answer_with_scores[i][2]   if answer_with_scores[i][2]   > 0 else 0.0 for i in valid_indices]
                n = len(valid_indices)
                tw_n_list     = [think_with_scores[i][2] for i in valid_indices]
                think_deltas  = [tw_avg[j] - tow_avg[j]  for j in range(n)]
                # Post-regularization think delta (clip + length floor) — what
                # actually enters the reward. Raw think_delta_mean is kept
                # alongside so the gap exposes how much the guards are biting.
                think_deltas_reg = [self._reg_think_delta(think_deltas[j], tw_n_list[j]) for j in range(n)]
                answer_deltas = [aw_avg[j] - aow_avg[j]  for j in range(n)]
                final_deltas  = [think_deltas[j] - self.beta * answer_deltas[j] for j in range(n)]

                metrics.update({
                    "reward/think_with_mean":     sum(tw_avg)  / n,
                    "reward/think_without_mean":  sum(tow_avg) / n,
                    "reward/answer_with_mean":    sum(aw_avg)  / n,
                    "reward/answer_without_mean": sum(aow_avg) / n,
                    "reward/think_delta_mean":    sum(think_deltas)  / n,
                    "reward/think_delta_reg_mean": sum(think_deltas_reg) / n,
                    "reward/answer_delta_mean":   sum(answer_deltas) / n,
                    "reward/final_delta_mean":    sum(final_deltas)  / n,
                    "reward/think_entropy_mean":  sum(tw_ent)  / n,
                    "reward/answer_entropy_mean": sum(aw_ent)  / n,
                })


            # --- attention ratio (always computed exactly once per rank) ---
            # Use the first valid sample if one exists; otherwise fall back to sample 0.
            # _plan_attention_score always calls model() even when spans are invalid, so
            # every rank does the same number of ZeRO-3 all-gathers regardless of whether
            # its batch slice contains any scorable examples.
            _attn_idx = valid_indices[0] if valid_indices else 0
            _parsed_i = parsed_results[_attn_idx]
            _prompt_i = prompts[_attn_idx]
            _text_i   = completions[_attn_idx] if isinstance(completions[_attn_idx], str) else ""
            _plan_span = _shift_span(_parsed_i["abstract"], len(_prompt_i)) if _parsed_i["abstract"] is not None else (0, 0)
            _ans_span  = _shift_span(_parsed_i["answer"],   len(_prompt_i)) if _parsed_i["answer"]   is not None else (0, 0)
            score = _plan_attention_score(self.model, self.tokenizer, _prompt_i + _text_i, _plan_span, _ans_span)
            # None is expected when this rank's batch slice has NO valid samples
            # (valid_indices empty → fell back to sample 0 with dummy (0,0) spans).
            # It can also occur on a "valid" sample whose plan/answer chars don't map
            # to any tokens (e.g. abstract collapsed to whitespace, tokenizer offset
            # mapping skipped). Treat as a missing metric — don't crash the run.
            if score is None and valid_indices:
                _log_stage("reward", (
                    f"WARN _plan_attention_score returned None at step={self._call_count} "
                    f"sample_idx={_attn_idx} plan_span={_plan_span} ans_span={_ans_span} "
                    f"— skipping attn metric this step"
                ))
            if score is not None:
                metrics["reward/attn_answer_to_plan"] = score

            # --- log to stdout ---
            _log_stage("reward", (
                f"base={metrics['reward/base_reward_mean']:.3f}"
                f"  valid={metrics['reward/valid_fraction']:.2f}"
                + (f"  think_delta={metrics['reward/think_delta_mean']:.4f}"
                   f"  answer_delta={metrics['reward/answer_delta_mean']:.4f}"
                   f"  final_delta={metrics['reward/final_delta_mean']:.4f}"
                   f"  H_think={metrics['reward/think_entropy_mean']:.3f}"
                   f"  H_answer={metrics['reward/answer_entropy_mean']:.3f}"
                   if "reward/think_delta_mean" in metrics else "  (no valid examples)")
                + (f"  attn={metrics['reward/attn_answer_to_plan']:.4f}"
                   if "reward/attn_answer_to_plan" in metrics else "")
            ))
            _log_stage("lengths", (
                f"abstract={metrics['lengths/abstract_chars_mean']:.1f}"
                f"  think={metrics['lengths/think_chars_mean']:.1f}  chars"
            ))

            # --- contrastive CoT: sample no-plan rollouts, compare answer quality ---
            contrastive_rewards: list[float | None] = [None] * len(completions)
            if self.contrastive_cot:
                if self.vllm_engine is not None:
                    no_plan_rollouts = self._generate_contrastive_completions_vllm(prompts)
                else:
                    no_plan_rollouts = self._generate_contrastive_completions(prompts)
                contrastive_base: list[float] = []
                for rollouts, gold, nums in zip(no_plan_rollouts, target, numbers):
                    per_rollout_rewards: list[float] = []
                    for no_plan_text in rollouts:
                        no_plan_parsed = process_response(no_plan_text)
                        no_plan_answer = _extract_text(no_plan_text, no_plan_parsed["answer"])
                        per_rollout_rewards.append(
                            _final_answer_reward(no_plan_answer, int(gold), [int(x) for x in nums])
                        )
                    contrastive_base.append(sum(per_rollout_rewards) / len(per_rollout_rewards))
                for idx in range(len(completions)):
                    contrastive_rewards[idx] = base_rewards[idx] - contrastive_base[idx]

                n_total = len(completions)
                metrics["contrastive/no_plan_reward_mean"] = sum(contrastive_base) / n_total
                metrics["contrastive/delta_mean"] = sum(
                    cr for cr in contrastive_rewards if cr is not None
                ) / n_total
                _log_stage("contrastive", (
                    f"with_plan_reward={metrics['reward/base_reward_mean']:.3f}"
                    f"  no_plan_reward={metrics['contrastive/no_plan_reward_mean']:.3f}"
                    f"  contrastive_delta={metrics['contrastive/delta_mean']:.3f}"
                ))

            # --- wandb ---
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({k: v for k, v in metrics.items() if k != "step"})
            except ImportError:
                pass

            # --- file (rank 0 only, always, every step) ---
            if self._metrics_file is not None:
                self._metrics_file.write(json.dumps(metrics) + "\n")
                self._metrics_file.flush()

            self._call_count += 1

            for idx, base_reward in enumerate(base_rewards):
                if not valid_examples[idx]:
                    rewards.append(self.base_reward_weight * base_reward)
                    continue

                tw_sum, _tw_ent, tw_n   = think_with_scores[idx]
                tow_sum, _tow_ent, tow_n = think_without_scores[idx]
                aw_sum, _aw_ent, aw_n    = answer_with_scores[idx]
                aow_sum, _aow_ent, aow_n = answer_without_scores[idx]

                tw_pt  = tw_sum  / tw_n  if tw_n  > 0 else 0.0
                tow_pt = tow_sum / tow_n if tow_n > 0 else 0.0
                aw_pt  = aw_sum  / aw_n  if aw_n  > 0 else 0.0
                aow_pt = aow_sum / aow_n if aow_n > 0 else 0.0

                think_d = self._reg_think_delta(tw_pt - tow_pt, tw_n)
                delta = self.think_delta_weight * think_d - self.beta * (aw_pt - aow_pt)

                # Add contrastive reward when enabled
                if contrastive_rewards[idx] is not None:
                    delta += self.contrastive_weight * contrastive_rewards[idx]

                _log_stage("reward", f"delta reward={delta:.4f} (per-token)")
                rewards.append(self.base_reward_weight * base_reward + delta)
        finally:
            if was_training:
                self.model.train()

        return rewards


class CountdownEvalCallback(TrainerCallback):
    """Run countdown accuracy evaluation every N training steps.

    Generates completions for a held-out eval set using the current model,
    evaluates answer correctness, and logs results to wandb / stdout / file.
    """

    def __init__(
        self,
        tokenizer: Any,
        eval_file: str | Path,
        output_dir: str | Path,
        every_n_steps: int = 10,
        max_samples: int = 100,
        max_new_tokens: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_file = Path(eval_file)
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self._eval_data: list[dict[str, Any]] | None = None

    def _load_eval_data(self) -> list[dict[str, Any]]:
        if self._eval_data is not None:
            return self._eval_data
        rows: list[dict[str, Any]] = []
        with self.eval_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                numbers, target = _parse_countdown_input(item["input"])
                rows.append({"numbers": numbers, "target": target})
                if len(rows) >= self.max_samples:
                    break
        self._eval_data = rows
        return rows

    def _build_prompt(self, numbers: list[int], target: int) -> str:
        problem_text = _build_problem_text(numbers, target)
        user_content = abstract_prompt.format(
            TASK_DESCIPTION=TASK_DESCRIPTION,
            PROBLEM_TEXT=problem_text,
        )
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def _run_eval(self, model: Any, global_step: int) -> dict[str, Any]:
        import os
        _is_main = os.environ.get("LOCAL_RANK", "0") == "0"

        _hf_model = model.module if hasattr(model, "module") else model
        was_training = _hf_model.training
        _hf_model.eval()
        device = next(_hf_model.parameters()).device

        eval_data = self._load_eval_data()
        correct = 0
        total = len(eval_data)
        results: list[dict[str, Any]] = []

        for row in eval_data:
            prompt = self._build_prompt(row["numbers"], row["target"])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = _hf_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            completion = self.tokenizer.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True
            ).strip()

            parsed = process_response(completion)
            answer_text = _extract_text(completion, parsed["answer"])
            reward = _final_answer_reward(answer_text, row["target"], row["numbers"])
            is_correct = reward == 1.0
            correct += int(is_correct)
            results.append({
                "target": row["target"],
                "numbers": row["numbers"],
                "answer_text": answer_text,
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0

        if was_training:
            _hf_model.train()

        summary = {
            "step": global_step,
            "countdown_eval/accuracy": accuracy,
            "countdown_eval/correct": correct,
            "countdown_eval/total": total,
        }

        if _is_main:
            _log_stage("countdown_eval", (
                f"step={global_step}  accuracy={accuracy:.4f}  ({correct}/{total})"
            ))

            # Write per-step eval results
            eval_dir = self.output_dir / "countdown_evals"
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_file = eval_dir / f"step_{global_step}.json"
            with eval_file.open("w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": results}, f, indent=2)

            # Log to wandb
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log(summary)
            except ImportError:
                pass

        return summary

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return
        if model is None:
            return
        self._run_eval(model, state.global_step)


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
        # DAPO / clip-higher / token-filter knobs (TRL GRPOConfig accepts these
        # for both loss_type="grpo" and "dapo"). Defaults preserve GRPO behavior.
        loss_type=args.loss_type,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        mask_truncated_completions=args.mask_truncated_completions,
        top_entropy_quantile=args.top_entropy_quantile,
        scale_rewards=args.scale_rewards,
        **build_grpo_vllm_kwargs(args),
        # deepspeed
        deepspeed=ds_cfg,
    )

    _log_stage("reward", "building countdown reward function")
    reward_fn = PlanAwareReward(
        tokenizer=tokenizer,
        beta=args.reward_beta,
        output_dir=args.output_dir,
        contrastive_cot=args.contrastive_cot,
        contrastive_weight=args.contrastive_weight,
        contrastive_max_tokens=args.contrastive_max_tokens,
        think_delta_weight=args.think_delta_weight,
        base_reward_weight=args.base_reward_weight,
        think_delta_clip=args.think_delta_clip,
        think_min_tokens=args.think_min_tokens,
    )
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
    reward_fn.bind_vllm_engine(getattr(trainer, "vllm_generation", None))

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
    parser.add_argument("--reward_beta", type=float, default=0.5,
                        help="Weight on the answer logprob delta (with-plan − without-plan). 0 disables the term.")
    parser.add_argument("--think_delta_weight", type=float, default=1.0,
                        help="Weight on the think-block logprob delta. 0 disables the term.")
    parser.add_argument("--think_delta_clip", type=float, default=0.0,
                        help="Symmetric clip on the raw think logprob delta before weighting "
                             "(0 = no clip). Bounds the term to ±clip so it cannot dominate the "
                             "base reward when the policy Goodharts per-token logprob.")
    parser.add_argument("--think_min_tokens", type=int, default=0,
                        help="Length floor for the think-delta term (0 = off). The term is "
                             "linearly ramped to 0 as the think span shrinks below this many "
                             "tokens, removing the incentive to collapse `think` length.")
    parser.add_argument("--base_reward_weight", type=float, default=1.0,
                        help="Weight on the base (correctness) reward. >1 emphasizes base over the auxiliary terms.")
    parser.add_argument("--contrastive_cot", action="store_true",
                        help="Enable contrastive CoT: generate a no-plan baseline and reward the quality gap.")
    parser.add_argument("--contrastive_weight", type=float, default=0.3,
                        help="Weight for the contrastive reward term.")
    parser.add_argument("--contrastive_max_tokens", type=int, default=2048,
                        help="Max new tokens for contrastive (no-plan) generation.")
    parser.add_argument("--countdown_eval_steps", type=int, default=0,
                        help="Run countdown accuracy eval every N steps (0 = disabled).")
    parser.add_argument("--countdown_eval_samples", type=int, default=100,
                        help="Number of eval samples for in-training countdown evaluation.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1)

    # DAPO / clip-higher / token-filter (defaults preserve plain GRPO behavior)
    parser.add_argument("--loss_type", choices=["grpo", "dapo"], default="grpo",
                        help="TRL loss type. 'dapo' enables token-level loss + clip-higher.")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Lower clip ratio (PPO-style). DAPO recipe: 0.2.")
    parser.add_argument("--epsilon_high", type=float, default=None,
                        help="Upper clip ratio for asymmetric clipping. DAPO recipe: 0.28.")
    parser.add_argument("--mask_truncated_completions", action="store_true",
                        help="Drop rollouts that hit max_completion_length from the loss (DAPO).")
    parser.add_argument("--top_entropy_quantile", type=float, default=1.0,
                        help="Train only on tokens whose entropy is in the top quantile. "
                             "1.0 = all tokens (GRPO default); DAPO recipe: 0.2.")
    parser.add_argument("--scale_rewards", choices=["group", "batch", "none"], default="group",
                        help="Reward standardization scope.")
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
