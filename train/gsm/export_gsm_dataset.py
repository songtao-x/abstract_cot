from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from .script.gsm_utils import (
        build_export_row,
        generate_gsm_problem,
        seed_gsm_generation,
    )
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent / "script"
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gsm_utils import (  # type: ignore
        build_export_row,
        generate_gsm_problem,
        seed_gsm_generation,
    )


DEFAULT_OPS = [16, 17, 18, 19, 20, 21, 22]
DEFAULT_CONDITIONS = ["light", "medium", "hard"]
DEFAULT_SMOKE_TRAIN_FILE = Path(__file__).resolve().parents[2] / "data" / "gsm_smoke_train.jsonl"
DEFAULT_SMOKE_EVAL_FILE = Path(__file__).resolve().parents[2] / "data" / "gsm_smoke_eval.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local GSM-DC smoke dataset for RL experiments.")
    parser.add_argument("--train_out", type=str, default=str(DEFAULT_SMOKE_TRAIN_FILE))
    parser.add_argument("--eval_out", type=str, default=str(DEFAULT_SMOKE_EVAL_FILE))
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_rows(total_rows: int, seed: int) -> list[dict]:
    seed_gsm_generation(seed)
    rows: list[dict] = []
    pairs = [(op, condition) for op in DEFAULT_OPS for condition in DEFAULT_CONDITIONS]
    for idx in range(total_rows):
        op, condition = pairs[idx % len(pairs)]
        problem = generate_gsm_problem(op=op, condition=condition)
        rows.append(build_export_row(problem=problem, condition=condition, op=op))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    train_rows = generate_rows(args.train_size, args.seed)
    eval_rows = generate_rows(args.eval_size, args.seed + 1)

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    print(f"Wrote {len(train_rows)} training rows to {train_out}")
    print(f"Wrote {len(eval_rows)} eval rows to {eval_out}")


if __name__ == "__main__":
    main()
