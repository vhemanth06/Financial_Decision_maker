from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


# Generate CPCV splits with index-based purging around test windows.
@dataclass(frozen=True)
class CombinatorialPurgedCV:
    n_splits: int
    n_test_splits: int
    purge_window: int

    # Validate split hyperparameters.
    def __post_init__(self) -> None:
        if self.n_splits <= 1:
            raise ValueError("n_splits must be greater than 1")
        if self.n_test_splits < 1:
            raise ValueError("n_test_splits must be at least 1")
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits must be smaller than n_splits")
        if self.purge_window < 0:
            raise ValueError("purge_window must be non-negative")

    # Return the number of combinatorial train/test configurations.
    def get_n_splits(self) -> int:
        combo_count = len(list(combinations(range(self.n_splits), self.n_test_splits)))
        return combo_count

    # Split chronological sample indices into contiguous folds.
    def _build_fold_indices(self, n_samples: int) -> list[np.ndarray]:
        full_index = np.arange(n_samples, dtype=np.int64)
        return [np.array(part, dtype=np.int64) for part in np.array_split(full_index, self.n_splits)]

    # Remove train rows that are too close to any test row.
    def _purge_indices(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        if self.purge_window == 0:
            return train_indices

        exclusion_mask = np.zeros(n_samples, dtype=bool)
        for test_idx in test_indices:
            start = max(0, int(test_idx) - self.purge_window)
            stop = min(n_samples, int(test_idx) + self.purge_window + 1)
            exclusion_mask[start:stop] = True

        return train_indices[~exclusion_mask[train_indices]]

    # Yield purged CPCV train/test index pairs.
    def split(self, data: Sequence[Any]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(data)
        fold_indices = self._build_fold_indices(n_samples)

        fold_ids = list(range(self.n_splits))
        for test_fold_combo in combinations(fold_ids, self.n_test_splits):
            test_indices = np.concatenate([fold_indices[fold_id] for fold_id in test_fold_combo])

            train_fold_ids = [fold_id for fold_id in fold_ids if fold_id not in test_fold_combo]
            if not train_fold_ids:
                continue

            train_indices = np.concatenate([fold_indices[fold_id] for fold_id in train_fold_ids])
            train_indices = self._purge_indices(train_indices, test_indices, n_samples)

            if train_indices.size == 0:
                continue

            yield np.sort(train_indices), np.sort(test_indices)


# Load project configuration from YAML.
def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


# Instantiate CombinatorialPurgedCV using config values.
def build_cpcv_from_config(config: dict[str, Any]) -> CombinatorialPurgedCV:
    cpcv_cfg = config["cpcv"]
    return CombinatorialPurgedCV(
        n_splits=cpcv_cfg["n_splits"],
        n_test_splits=cpcv_cfg["n_test_splits"],
        purge_window=cpcv_cfg["purge_window"],
    )


# Parse command-line arguments for standalone validation.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CPCV split counts.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    parser.add_argument("--sample-count", required=True, type=int, help="Number of samples")
    return parser.parse_args()


# Log the number of generated CPCV splits for a sample size.
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    splitter = build_cpcv_from_config(config)
    generated = sum(1 for _ in splitter.split(list(range(args.sample_count))))
    LOGGER.info("Generated %d CPCV splits", generated)


if __name__ == "__main__":
    main()
