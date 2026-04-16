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


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """Generate CPCV splits with index-based purging around test windows.

    CPCV is used to reduce leakage from nearby samples when features include rolling
    statistics, making validation outcomes more reliable for financial time series.
    """

    n_splits: int
    n_test_splits: int
    purge_window: int

    def __post_init__(self) -> None:
        """Validate split hyperparameters at object construction."""
        if self.n_splits <= 1:
            raise ValueError("n_splits must be greater than 1")
        if self.n_test_splits < 1:
            raise ValueError("n_test_splits must be at least 1")
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits must be smaller than n_splits")
        if self.purge_window < 0:
            raise ValueError("purge_window must be non-negative")

    def get_n_splits(self) -> int:
        """Return the number of combinatorial train/test configurations."""
        combo_count = len(list(combinations(range(self.n_splits), self.n_test_splits)))
        return combo_count

    def _build_fold_indices(self, n_samples: int) -> list[np.ndarray]:
        """Split chronological sample indices into contiguous folds.

        Args:
            n_samples: Number of rows in the dataset.

        Returns:
            List of contiguous index arrays.
        """
        full_index = np.arange(n_samples, dtype=np.int64)
        return [np.array(part, dtype=np.int64) for part in np.array_split(full_index, self.n_splits)]

    def _purge_indices(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Remove train rows that are too close to any test row.

        Args:
            train_indices: Candidate training indices.
            test_indices: Test indices.
            n_samples: Total sample count.

        Returns:
            Purged training indices.
        """
        if self.purge_window == 0:
            return train_indices

        exclusion_mask = np.zeros(n_samples, dtype=bool)
        for test_idx in test_indices:
            start = max(0, int(test_idx) - self.purge_window)
            stop = min(n_samples, int(test_idx) + self.purge_window + 1)
            exclusion_mask[start:stop] = True

        return train_indices[~exclusion_mask[train_indices]]

    def split(self, data: Sequence[Any]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield purged CPCV train/test index pairs.

        Args:
            data: Sequence-like dataset used only for length inference.

        Yields:
            Tuples of (train_indices, test_indices).
        """
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


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def build_cpcv_from_config(config: dict[str, Any]) -> CombinatorialPurgedCV:
    """Instantiate CombinatorialPurgedCV using config values.

    Args:
        config: Full project configuration dictionary.

    Returns:
        Configured CombinatorialPurgedCV object.
    """
    cpcv_cfg = config["cpcv"]
    return CombinatorialPurgedCV(
        n_splits=cpcv_cfg["n_splits"],
        n_test_splits=cpcv_cfg["n_test_splits"],
        purge_window=cpcv_cfg["purge_window"],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone validation."""
    parser = argparse.ArgumentParser(description="Inspect CPCV split counts.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    parser.add_argument("--sample-count", required=True, type=int, help="Number of samples")
    return parser.parse_args()


def main() -> None:
    """Log the number of generated CPCV splits for a sample size."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    splitter = build_cpcv_from_config(config)
    generated = sum(1 for _ in splitter.split(list(range(args.sample_count))))
    LOGGER.info("Generated %d CPCV splits", generated)


if __name__ == "__main__":
    main()
