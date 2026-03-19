from __future__ import annotations

import os
import re
from typing import Callable


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _safe_rank(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _is_primary_process() -> bool:
    rank = _safe_rank(os.getenv("RANK"))
    if rank is not None:
        return rank == 0
    slurm_rank = _safe_rank(os.getenv("SLURM_PROCID"))
    if slurm_rank is not None:
        return slurm_rank == 0
    return True


def _normalize_component(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def _model_slug(model_name: str) -> str:
    tail = model_name.rsplit("/", 1)[-1] if "/" in model_name else model_name
    return _normalize_component(tail)


def build_wandb_run_name(
    task: str,
    loss_type: str,
    reward_variant: str,
    model_name: str,
) -> str:
    return "-".join(
        [
            _normalize_component(task),
            _normalize_component(loss_type),
            _normalize_component(reward_variant),
            _model_slug(model_name),
        ]
    )


def setup_wandb(
    task: str,
    loss_type: str,
    reward_variant: str,
    model_name: str,
    wandb_project: str,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> str:
    project = (wandb_project or "").strip()
    entity = (wandb_entity or "").strip()
    run_name = (wandb_run_name or "").strip() or build_wandb_run_name(
        task=task,
        loss_type=loss_type,
        reward_variant=reward_variant,
        model_name=model_name,
    )

    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    else:
        os.environ.pop("WANDB_ENTITY", None)
    os.environ["WANDB_NAME"] = run_name

    preset_mode = (os.getenv("WANDB_MODE") or "").strip().lower()
    if preset_mode in {"offline", "disabled", "dryrun"}:
        _log(log_fn, f"using preconfigured WANDB_MODE={preset_mode} run_name={run_name}")
        return run_name

    os.environ["WANDB_MODE"] = "online"
    if not _is_primary_process():
        return run_name

    try:
        import wandb

        if getattr(wandb, "run", None) is None:
            wandb.init(
                project=project,
                entity=entity or None,
                name=run_name,
                mode="online",
                reinit=False,
            )
        _log(log_fn, f"initialized online logging run_name={run_name}")
    except Exception as exc:
        os.environ["WANDB_MODE"] = "offline"
        _log(log_fn, f"online init failed ({exc}); switching to WANDB_MODE=offline")
        try:
            import wandb

            if getattr(wandb, "run", None) is None:
                wandb.init(
                    project=project,
                    entity=entity or None,
                    name=run_name,
                    mode="offline",
                    reinit=False,
                )
        except Exception as offline_exc:
            os.environ["WANDB_DISABLED"] = "true"
            _log(log_fn, f"offline init failed ({offline_exc}); disabling wandb for this run")

    return run_name
