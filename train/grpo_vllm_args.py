from __future__ import annotations

from typing import Any


def build_grpo_vllm_kwargs(args: Any) -> dict[str, Any]:
    return {
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_model_impl": args.vllm_model_impl,
        "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
        "vllm_server_base_url": args.vllm_server_base_url,
        "vllm_server_host": args.vllm_server_host,
        "vllm_server_port": args.vllm_server_port,
        "vllm_server_timeout": args.vllm_server_timeout,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_length": args.vllm_max_model_length,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
    }
