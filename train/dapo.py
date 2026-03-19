from __future__ import annotations

from pathlib import Path

try:
    from .train_rl import main as run_main
except ImportError:
    from train_rl import main as run_main


def main() -> None:
    run_main(
        default_overrides={
            "loss_type": "dapo",
            "epsilon": 0.2,
            "epsilon_high": 0.28,
            "mask_truncated_completions": True,
            "top_entropy_quantile": 0.2,
            "scale_rewards": "group",
            "use_vllm": True,
            "vllm_mode": "colocate",
            "output_dir": str(Path(__file__).resolve().parent / "outputs" / "dapo_task"),
        },
        description="Train task-aware DAPO-profile experiments on countdown or GSM data.",
    )


if __name__ == "__main__":
    main()
