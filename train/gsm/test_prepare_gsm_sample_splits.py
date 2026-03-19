import builtins
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent / "script"
TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TRAIN_DIR))


class BuildExportRowFromSampleTest(unittest.TestCase):
    def test_normalizes_condition_label_suffixes(self) -> None:
        sys.modules.pop("gsm_utils", None)
        from gsm_utils import build_export_row_from_sample

        row = build_export_row_from_sample(
            {
                "problem_text": "Test problem",
                "problem_json": {"problem_info": {"final_answer": 5}, "problem_text": []},
                "final_answer": 5,
                "condition": "condition_3",
                "op": 2,
            },
            row_index=0,
        )

        self.assertEqual(row["condition"], "hard")


class PrepareGsmSampleSplitsImportTest(unittest.TestCase):
    def test_prepare_script_import_does_not_require_heavy_generation_dependencies(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".", 1)[0]
            if root in {"networkx", "numpy", "torch"}:
                raise AssertionError(f"unexpected heavy import: {name}")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guarded_import):
            sys.modules.pop("prepare_gsm_sample_splits", None)
            sys.modules.pop("gsm_utils", None)
            module = importlib.import_module("prepare_gsm_sample_splits")

        self.assertTrue(hasattr(module, "stratify_rows"))


class LoadHfRowsTest(unittest.TestCase):
    def test_retries_without_data_files_when_cached_config_differs(self) -> None:
        sys.modules.pop("prepare_gsm_sample_splits", None)
        module = importlib.import_module("prepare_gsm_sample_splits")

        calls: list[tuple[str, str | None]] = []

        def fake_load_dataset(dataset_name, data_files=None):
            calls.append((dataset_name, data_files))
            if data_files is not None:
                raise ValueError(
                    "Couldn't find cache for YMinglai/GSM-DC-Dataset-Sample for config "
                    "'default-data_files=all_problems.json'\nAvailable configs in the cache: ['default-31f2f1efbf49a232']"
                )
            return {"train": [{"problem_text": "cached row"}]}

        fake_datasets = ModuleType("datasets")
        fake_datasets.load_dataset = fake_load_dataset

        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            rows = module.load_hf_rows("YMinglai/GSM-DC-Dataset-Sample", "all_problems.json")

        self.assertEqual(rows, [{"problem_text": "cached row"}])
        self.assertEqual(
            calls,
            [
                ("YMinglai/GSM-DC-Dataset-Sample", "all_problems.json"),
                ("YMinglai/GSM-DC-Dataset-Sample", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
