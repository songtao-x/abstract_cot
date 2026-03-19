from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

try:
    from .script.gsm_utils import (
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
        DEFAULT_GSM_EVAL_FILE,
        DEFAULT_GSM_HF_DATA_FILE,
        DEFAULT_GSM_HF_DATASET,
        DEFAULT_GSM_SAMPLE_SOURCE_FILE,
        DEFAULT_GSM_TEST_FILE,
        DEFAULT_GSM_TRAIN_FILE,
        build_export_row_from_sample,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple GSM split pipeline: load HF (or local source), save raw rows locally, "
            "then split and process into train/valid/test JSONL."
        )
    )
    parser.add_argument("--source_json", type=str, default=None, help="Optional local raw source JSON/JSONL.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_GSM_HF_DATASET)
    parser.add_argument("--data_file", type=str, default=DEFAULT_GSM_HF_DATA_FILE)
    parser.add_argument(
        "--raw_cache_json",
        type=str,
        default=str(DEFAULT_GSM_SAMPLE_SOURCE_FILE),
        help="Where to save raw rows loaded from HF or source_json.",
    )
    parser.add_argument("--train_out", type=str, default=str(DEFAULT_GSM_TRAIN_FILE))
    parser.add_argument("--valid_out", type=str, default=str(DEFAULT_GSM_EVAL_FILE))
    parser.add_argument("--test_out", type=str, default=str(DEFAULT_GSM_TEST_FILE))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
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
        return rows

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "rows", "train", "problems"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(
        f"Unsupported source format in {path}; expected JSON array, JSONL, or object with data/rows/train/problems."
    )


def load_hf_rows(dataset_name: str, data_file: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load from Hugging Face") from exc

    ds_obj: Any
    try:
        ds_obj = load_dataset(dataset_name, data_files=data_file)
        print(
            f"[prepare_gsm_sample_splits_simple] source=hf dataset={dataset_name} "
            f"data_file={data_file} mode=with_data_file"
        )
    except Exception as first_exc:
        try:
            ds_obj = load_dataset(dataset_name)
            print(
                f"[prepare_gsm_sample_splits_simple] source=hf dataset={dataset_name} "
                "mode=default_config_fallback"
            )
            print(
                "[prepare_gsm_sample_splits_simple] note=failed_data_file_load "
                f"error={type(first_exc).__name__}: {first_exc}"
            )
        except Exception as second_exc:
            raise RuntimeError(
                "Failed to load HF dataset with data_file and default config.\n"
                f"with_data_file_error: {type(first_exc).__name__}: {first_exc}\n"
                f"default_config_error: {type(second_exc).__name__}: {second_exc}"
            ) from second_exc

    if "train" in ds_obj:
        split = ds_obj["train"]
    else:
        split_name = next(iter(ds_obj.keys()))
        split = ds_obj[split_name]
        print(f"[prepare_gsm_sample_splits_simple] using_split={split_name}")
    return [dict(row) for row in split]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in [0,1), got {valid_ratio}")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.0")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    return shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]


def process_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        processed.append(build_export_row_from_sample(row, i))
    return processed


def main() -> None:
    args = parse_args()

    if args.source_json:
        source_path = Path(args.source_json)
        if not source_path.exists():
            print(f"source_json not found: {source_path}", file=sys.stderr)
            raise SystemExit(1)
        raw_rows = load_source_rows(source_path)
        print(f"[prepare_gsm_sample_splits_simple] source=local path={source_path} rows={len(raw_rows)}")
    else:
        raw_rows = load_hf_rows(args.dataset_name, args.data_file)
        print(f"[prepare_gsm_sample_splits_simple] source=hf rows={len(raw_rows)}")

    if not raw_rows:
        raise ValueError("No rows loaded.")

    raw_cache_path = Path(args.raw_cache_json)
    write_json(raw_cache_path, raw_rows)
    print(f"[prepare_gsm_sample_splits_simple] wrote_raw_cache={raw_cache_path}")

    processed_rows = process_rows(raw_rows)
    train_rows, valid_rows, test_rows = split_rows(
        processed_rows,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    train_out = Path(args.train_out)
    valid_out = Path(args.valid_out)
    test_out = Path(args.test_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(valid_out, valid_rows)
    write_jsonl(test_out, test_rows)

    print(f"[prepare_gsm_sample_splits_simple] wrote_train={train_out} rows={len(train_rows)}")
    print(f"[prepare_gsm_sample_splits_simple] wrote_valid={valid_out} rows={len(valid_rows)}")
    print(f"[prepare_gsm_sample_splits_simple] wrote_test={test_out} rows={len(test_rows)}")


if __name__ == "__main__":
    main()
