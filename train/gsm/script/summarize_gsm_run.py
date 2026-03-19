from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a GSM training run.")
    parser.add_argument("--run_dir", type=str, required=True)
    return parser.parse_args()


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                step = int(path.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_last_log(trainer_state: dict[str, Any]) -> dict[str, Any]:
    history = trainer_state.get("log_history", [])
    if not history:
        return {}
    return history[-1]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    trainer_state_path = run_dir / "trainer_state.json"
    gsm_eval_path = run_dir / "gsm_eval.json"
    summary_json_path = run_dir / "summary.json"
    summary_txt_path = run_dir / "summary.txt"

    trainer_state = read_json(trainer_state_path) if trainer_state_path.exists() else {}
    gsm_eval = read_json(gsm_eval_path) if gsm_eval_path.exists() else {}
    latest_checkpoint = find_latest_checkpoint(run_dir)
    last_log = extract_last_log(trainer_state)
    eval_summary = gsm_eval.get("summary", {})

    summary = {
        "run_dir": str(run_dir),
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "global_step": trainer_state.get("global_step"),
        "best_metric": trainer_state.get("best_metric"),
        "last_log": last_log,
        "gsm_eval": eval_summary,
    }

    lines = [
        f"Run: {run_dir}",
        f"Latest checkpoint: {summary['latest_checkpoint']}",
        f"Global step: {summary['global_step']}",
        f"Best metric: {summary['best_metric']}",
        f"Last log: {json.dumps(last_log, ensure_ascii=True)}",
        f"GSM eval: {json.dumps(eval_summary, ensure_ascii=True)}",
    ]

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    with summary_txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_txt_path}")


if __name__ == "__main__":
    main()
