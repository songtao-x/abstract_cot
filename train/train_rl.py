from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from .cuda_alloc_conf import sanitize_pytorch_cuda_alloc_conf_for_vllm
    from .grpo_vllm_args import build_grpo_vllm_kwargs
    from .task_data import get_default_eval_file, get_default_train_file, load_task_dataset
    from .task_rewards import build_reward
    from .wandb_utils import setup_wandb
except ImportError:
    from cuda_alloc_conf import sanitize_pytorch_cuda_alloc_conf_for_vllm
    from grpo_vllm_args import build_grpo_vllm_kwargs
    from task_data import get_default_eval_file, get_default_train_file, load_task_dataset
    from task_rewards import build_reward
    from wandb_utils import setup_wandb

TRAIN_DIR = Path(__file__).resolve().parent
DEFAULT_DS_CFG = TRAIN_DIR / "ds_zero3.json"


def _log_stage(stage: str, message: str) -> None:
    print(f"[train_rl:{stage}] {message}", flush=True)


def build_arg_parser(
    description: str,
    default_overrides: dict[str, Any] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", choices=["countdown", "gsm"], default="countdown")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

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
    parser.add_argument("--reward_variant", choices=["plan", "pure"], default="plan")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1)

    parser.add_argument("--loss_type", choices=["grpo", "dapo"], default="grpo")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--mask_truncated_completions", action="store_true")
    parser.add_argument("--top_entropy_quantile", type=float, default=1.0)
    parser.add_argument("--scale_rewards", choices=["group", "batch", "none"], default="group")

    parser.add_argument("--use_vllm", dest="use_vllm", action="store_true")
    parser.add_argument("--no_use_vllm", dest="use_vllm", action="store_false")
    parser.set_defaults(use_vllm=False)
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

    parser.add_argument("--ds_cfg", type=str, default=str(DEFAULT_DS_CFG))
    parser.add_argument("--wandb_project", type=str, default="abstract_grpo_runs")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    if default_overrides:
        parser.set_defaults(**default_overrides)
    return parser


def parse_args(
    description: str,
    default_overrides: dict[str, Any] | None = None,
) -> argparse.Namespace:
    parser = build_arg_parser(description=description, default_overrides=default_overrides)
    args = parser.parse_args()
    if args.train_file is None:
        args.train_file = str(get_default_train_file(args.task))
    if args.eval_file is None:
        args.eval_file = str(get_default_eval_file(args.task))
    if args.output_dir is None:
        args.output_dir = str(TRAIN_DIR / "outputs" / f"{args.loss_type}_{args.task}")
    return args


def train(args: argparse.Namespace) -> None:
    _log_stage("start", f"task={args.task} model={args.model_name}")
    run_name = setup_wandb(
        task=args.task,
        loss_type=args.loss_type,
        reward_variant=args.reward_variant,
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log_stage("model", "loading model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    _log_stage("data", f"loading train dataset from {args.train_file}")
    train_ds = load_task_dataset(args.task, args.train_file, max_samples=args.max_train_samples)
    _log_stage("data", f"train dataset size={len(train_ds)}")

    _log_stage("deepspeed", f"loading config from {args.ds_cfg}")
    with open(args.ds_cfg, "r", encoding="utf-8") as handle:
        ds_cfg = json.load(handle)
    ds_cfg["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    eval_ds = None
    eval_strategy = "no"
    if not args.no_eval and args.eval_file and Path(args.eval_file).exists():
        _log_stage("data", f"loading eval dataset from {args.eval_file}")
        eval_ds = load_task_dataset(args.task, args.eval_file, max_samples=args.max_eval_samples)
        eval_strategy = "steps"
        _log_stage("data", f"eval dataset size={len(eval_ds)}")
    else:
        _log_stage("data", "evaluation disabled or eval file missing")

    _log_stage(
        "config",
        (
            f"building GRPOConfig loss_type={args.loss_type} use_vllm={args.use_vllm} "
            f"vllm_mode={args.vllm_mode}"
        ),
    )
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
        loss_type=args.loss_type,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        mask_truncated_completions=args.mask_truncated_completions,
        top_entropy_quantile=args.top_entropy_quantile,
        scale_rewards=args.scale_rewards,
        deepspeed=ds_cfg,
    )

    _log_stage(
        "reward",
        f"building reward function for task={args.task} variant={args.reward_variant}",
    )
    reward_fn = build_reward(
        args.task,
        tokenizer=tokenizer,
        beta=args.reward_beta,
        reward_variant=args.reward_variant,
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

    _log_stage("train", "starting trainer.train()")
    trainer.train()
    _log_stage("save", f"saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _log_stage("done", "training pipeline finished")


def main(
    default_overrides: dict[str, Any] | None = None,
    description: str = "Train task-aware GRPO/DAPO experiments.",
) -> None:
    args = parse_args(description=description, default_overrides=default_overrides)
    if args.bf16 and args.fp16:
        raise ValueError("Use either --bf16 or --fp16, not both.")
    train(args)
