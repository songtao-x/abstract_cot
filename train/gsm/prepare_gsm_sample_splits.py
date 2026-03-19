from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

try:
    from .script.gsm_utils import (
        CONDITION_ORDER,
        DEFAULT_GSM_EVAL_FILE,
        DEFAULT_GSM_HF_DATA_FILE,
        DEFAULT_GSM_HF_DATASET,
        DEFAULT_GSM_SAMPLE_SOURCE_FILE,
        DEFAULT_GSM_TEST_FILE,
        DEFAULT_GSM_TRAIN_FILE,
        build_export_row_from_sample,
    )
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent / "script"
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gsm_utils import (  # type: ignore
        CONDITION_ORDER,
        DEFAULT_GSM_EVAL_FILE,
        DEFAULT_GSM_HF_DATA_FILE,
        DEFAULT_GSM_HF_DATASET,
        DEFAULT_GSM_SAMPLE_SOURCE_FILE,
        DEFAULT_GSM_TEST_FILE,
        DEFAULT_GSM_TRAIN_FILE,
        build_export_row_from_sample,
    )


TRAIN_BASE = 31
TRAIN_EXTRA = 47
VALID_BASE = 7
VALID_EXTRA = 59
TEST_BASE = 7
TEST_EXTRA = 59
OPS = tuple(range(2, 23))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic train/valid/test splits from the GSM-DC 6.3k sample dataset."
    )
    parser.add_argument(
        "--source_json",
        type=str,
        default=None,
        help=(
            "Optional local override for all_problems.json. If omitted, the script loads the sample dataset from "
            "Hugging Face."
        ),
    )
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_GSM_HF_DATASET)
    parser.add_argument("--data_file", type=str, default=DEFAULT_GSM_HF_DATA_FILE)
    parser.add_argument("--train_out", type=str, default=str(DEFAULT_GSM_TRAIN_FILE))
    parser.add_argument("--valid_out", type=str, default=str(DEFAULT_GSM_EVAL_FILE))
    parser.add_argument("--test_out", type=str, default=str(DEFAULT_GSM_TEST_FILE))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError(f"Source dataset is empty: {path}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No rows found in JSONL source: {path}")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Source JSONL must contain objects only: {path}")
        return rows

    if isinstance(payload, list):
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"Source JSON array must contain objects only: {path}")
        return payload

    if isinstance(payload, dict):
        for key in ("data", "rows", "train", "problems"):
            value = payload.get(key)
            if isinstance(value, list):
                if not all(isinstance(row, dict) for row in value):
                    raise ValueError(f"Source JSON field '{key}' must contain objects only: {path}")
                return value

    raise ValueError(
        f"Unsupported source dataset format in {path}; expected a JSON array of rows, JSONL, or an object containing "
        "a row list under data/rows/train/problems."
    )


def load_hf_rows(dataset_name: str, data_file: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load the GSM-DC Hugging Face sample dataset") from exc

    try:
        dataset = load_dataset(dataset_name, data_files=data_file)["train"]
    except ValueError as exc:
        message = str(exc)
        if "Couldn't find cache" not in message or "Available configs in the cache" not in message:
            raise
        dataset = load_dataset(dataset_name)["train"]
    return [dict(row) for row in dataset]


def iter_strata() -> list[tuple[int, str]]:
    return [(op, condition) for op in OPS for condition in CONDITION_ORDER]


def build_allocation(index: int) -> tuple[int, int, int]:
    train_n = TRAIN_BASE + int(index < TRAIN_EXTRA)
    valid_n = VALID_BASE + int(index < VALID_EXTRA)
    test_n = TEST_BASE + int(index < TEST_EXTRA)
    return train_n, valid_n, test_n


def stratify_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strata = iter_strata()
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {key: [] for key in strata}

    for row_index, raw_row in enumerate(rows):
        converted = build_export_row_from_sample(raw_row, row_index)
        key = (int(converted["op"]), str(converted["condition"]))
        if key not in grouped:
            raise ValueError(
                f"Row {row_index} maps to unexpected stratum {key}; expected op in 2..22 and condition in {CONDITION_ORDER}"
            )
        grouped[key].append(converted)

    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for index, key in enumerate(strata):
        shard = list(grouped[key])
        train_n, valid_n, test_n = build_allocation(index)
        required = train_n + valid_n + test_n
        if len(shard) < required:
            raise ValueError(f"Stratum {key} has {len(shard)} rows, but {required} are required for the configured split")

        rng = random.Random((seed * 1000) + (key[0] * 10) + CONDITION_ORDER.index(key[1]))
        rng.shuffle(shard)

        train_rows.extend(shard[:train_n])
        valid_start = train_n
        valid_end = valid_start + valid_n
        valid_rows.extend(shard[valid_start:valid_end])
        test_end = valid_end + test_n
        test_rows.extend(shard[valid_end:test_end])

    return train_rows, valid_rows, test_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.source_json:
        source_json = Path(args.source_json)
        if not source_json.exists():
            print(f"Local GSM sample dataset override not found at {source_json}.", file=sys.stderr)
            raise SystemExit(1)
        print(f"[prepare_gsm_sample_splits] stage=load_source source={source_json}")
        rows = load_source_rows(source_json)
    else:
        print(
            f"[prepare_gsm_sample_splits] stage=load_hf_dataset dataset={args.dataset_name} data_file={args.data_file}"
        )
        rows = load_hf_rows(args.dataset_name, args.data_file)

    print(f"[prepare_gsm_sample_splits] stage=convert total_rows={len(rows)}")
    train_rows, valid_rows, test_rows = stratify_rows(rows, seed=args.seed)

    train_out = Path(args.train_out)
    valid_out = Path(args.valid_out)
    test_out = Path(args.test_out)

    print(f"[prepare_gsm_sample_splits] stage=write train={train_out} valid={valid_out} test={test_out}")
    write_jsonl(train_out, train_rows)
    write_jsonl(valid_out, valid_rows)
    write_jsonl(test_out, test_rows)

    print(f"[prepare_gsm_sample_splits] wrote train_rows={len(train_rows)}")
    print(f"[prepare_gsm_sample_splits] wrote valid_rows={len(valid_rows)}")
    print(f"[prepare_gsm_sample_splits] wrote test_rows={len(test_rows)}")


if __name__ == "__main__":
    main()
