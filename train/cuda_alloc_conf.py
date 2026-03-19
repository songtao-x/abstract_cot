from __future__ import annotations

import os
from collections.abc import MutableMapping


def _is_expandable_segments_true(entry: str) -> bool:
    key, sep, value = entry.partition(":")
    return (
        sep == ":"
        and key.strip().lower() == "expandable_segments"
        and value.strip().lower() == "true"
    )


def _remove_expandable_segments_true(conf: str) -> str:
    entries = [entry.strip() for entry in conf.split(",") if entry.strip()]
    sanitized = [entry for entry in entries if not _is_expandable_segments_true(entry)]
    return ",".join(sanitized)


def sanitize_pytorch_cuda_alloc_conf_for_vllm(
    use_vllm: bool,
    env: MutableMapping[str, str] | None = None,
) -> str | None:
    if env is None:
        env = os.environ

    conf = env.get("PYTORCH_CUDA_ALLOC_CONF")
    if conf is None or not use_vllm:
        return conf

    sanitized = _remove_expandable_segments_true(conf)
    if sanitized:
        env["PYTORCH_CUDA_ALLOC_CONF"] = sanitized
        return sanitized

    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return None
