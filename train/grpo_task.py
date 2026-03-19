from __future__ import annotations

from pathlib import Path

try:
    from .train_rl import main as run_main
except ImportError:
    from train_rl import main as run_main


def main() -> None:
    run_main(
        default_overrides={
            "loss_type": "grpo",
            "epsilon": 0.2,
            "epsilon_high": None,
            "mask_truncated_completions": False,
            "top_entropy_quantile": 1.0,
            "scale_rewards": "group",
        },
        description="Train task-aware GRPO experiments on countdown or GSM data.",
    )


if __name__ == "__main__":
    main()
