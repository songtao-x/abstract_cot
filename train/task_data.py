from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from datasets import Dataset

try:
    from .gsm.script.gsm_prompt_template import gsm_abstract_prompt
    from .gsm.script.gsm_utils import DEFAULT_GSM_EVAL_FILE, DEFAULT_GSM_TRAIN_FILE
    from .prompt_template import abstract_prompt
except ImportError:
    TRAIN_DIR = Path(__file__).resolve().parent
    GSM_SCRIPT_DIR = TRAIN_DIR / "gsm" / "script"
    for candidate in (TRAIN_DIR, GSM_SCRIPT_DIR):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    from gsm_prompt_template import gsm_abstract_prompt
    from gsm_utils import DEFAULT_GSM_EVAL_FILE, DEFAULT_GSM_TRAIN_FILE
    from prompt_template import abstract_prompt

TASK_DESCRIPTION = (
    "Countdown arithmetic task: use the provided numbers to reach the target with +, -, *, /. "
    "Use each provided number exactly once and provide a valid solution in the required tag format."
)

GSM_TASK_DESCRIPTION = (
    "Solve the grade-school math word problem exactly. Keep the abstract generic, write the detailed reasoning in a "
    "deterministic arithmetic-sentence format, and provide the final numeric answer only in the answer tag."
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_COUNTDOWN_TRAIN_FILE = DATA_DIR / "cd4_train.jsonl"
DEFAULT_COUNTDOWN_EVAL_FILE = DATA_DIR / "cd4_eval.jsonl"


def _parse_countdown_input(raw_input: str) -> tuple[list[int], int]:
    values = [int(x.strip()) for x in raw_input.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError(f"Invalid countdown input: {raw_input}")
    return values[:-1], values[-1]


def _build_countdown_problem_text(numbers: list[int], target: int) -> str:
    numbers_text = ", ".join(str(x) for x in numbers)
    return (
        f"Numbers: {numbers_text}\n"
        f"Target: {target}\n"
        "Find a valid expression that reaches the target using each listed number exactly once."
    )


def _build_gsm_problem_text(problem_text: str) -> str:
    return problem_text.strip()


def load_countdown_dataset(data_path: str | Path, max_samples: int | None = None) -> Dataset:
    path = Path(data_path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            numbers, target = _parse_countdown_input(item["input"])
            prompt = abstract_prompt.format(
                TASK_DESCIPTION=TASK_DESCRIPTION,
                PROBLEM_TEXT=_build_countdown_problem_text(numbers, target),
            )
            rows.append(
                {
                    "prompt": prompt,
                    "task_name": "countdown",
                    "target": int(target),
                    "numbers": numbers,
                    "problem_json": "",
                    "reference": item.get("output", ""),
                    "metadata": "",
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"No valid samples found in {path}")

    return Dataset.from_list(rows)


def load_gsm_dataset(data_path: str | Path, max_samples: int | None = None) -> Dataset:
    path = Path(data_path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            prompt = gsm_abstract_prompt.format(
                TASK_DESCIPTION=GSM_TASK_DESCRIPTION,
                PROBLEM_TEXT=_build_gsm_problem_text(item["problem_text"]),
            )
            metadata = json.dumps(
                {
                    "condition": item.get("condition", ""),
                    "op": item.get("op"),
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            problem_json = item["problem_json"]
            if not isinstance(problem_json, str):
                problem_json = json.dumps(problem_json, ensure_ascii=True, separators=(",", ":"))

            rows.append(
                {
                    "prompt": prompt,
                    "task_name": "gsm",
                    "target": int(item["final_answer"]),
                    "numbers": [],
                    "problem_json": problem_json,
                    "reference": item.get("reference_solution", ""),
                    "metadata": metadata,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"No valid samples found in {path}")

    return Dataset.from_list(rows)


def get_default_train_file(task: str) -> Path:
    if task == "gsm":
        return DEFAULT_GSM_TRAIN_FILE
    return DEFAULT_COUNTDOWN_TRAIN_FILE


def get_default_eval_file(task: str) -> Path:
    if task == "gsm":
        return DEFAULT_GSM_EVAL_FILE
    return DEFAULT_COUNTDOWN_EVAL_FILE


def load_task_dataset(task: str, data_path: str | Path, max_samples: int | None = None) -> Dataset:
    if task == "gsm":
        return load_gsm_dataset(data_path, max_samples=max_samples)
    if task == "countdown":
        return load_countdown_dataset(data_path, max_samples=max_samples)
    raise ValueError(f"Unsupported task: {task}")
