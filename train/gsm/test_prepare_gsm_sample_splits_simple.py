import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))


class LoadHfRowsSimpleTest(unittest.TestCase):
    def test_retries_with_cached_config_name(self) -> None:
        sys.modules.pop("prepare_gsm_sample_splits_simple", None)
        module = importlib.import_module("prepare_gsm_sample_splits_simple")

        calls: list[tuple[str | None, str | None]] = []

        def fake_load_dataset(dataset_name, data_files=None, name=None):
            del dataset_name
            calls.append((data_files, name))
            if data_files is not None:
                raise ValueError(
                    "Couldn't find cache for YMinglai/GSM-DC-Dataset-Sample for config "
                    "'default-data_files=all_problems.json'\n"
                    "Available configs in the cache: ['default-31f2f1efbf49a232']"
                )
            if name is None:
                raise ValueError(
                    "Couldn't find cache for YMinglai/GSM-DC-Dataset-Sample\n"
                    "Available configs in the cache: ['default-31f2f1efbf49a232']"
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
                ("all_problems.json", None),
                (None, None),
                (None, "default-31f2f1efbf49a232"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
