from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def generate_targets_with_dynamic_threshold(
    frame: pd.DataFrame,
    asset_column: str,
    date_column: str,
    price_column: str,
    target_column: str,
    horizon_days: int,
    rolling_window: int,
    rolling_min_periods: int,
    vol_multiplier: float,
) -> pd.DataFrame:
    """Create return labels using a rolling volatility-dependent threshold.

    This function builds training targets aligned with market regime changes, which is
    more useful for Sharpe optimization than fixed thresholds under non-stationary volatility.

    Args:
        frame: Input dataframe containing at least asset/date/price columns.
        asset_column: Name of the asset identifier column.
        date_column: Name of the chronological ordering column.
        price_column: Name of the price column.
        target_column: Destination column for discrete class targets.
        horizon_days: Forecast horizon in days.
        rolling_window: Rolling window size for volatility estimation.
        rolling_min_periods: Minimum observations for rolling statistics.
        vol_multiplier: Multiplier applied to rolling volatility to produce tau.

    Returns:
        Dataframe with added columns: future_price_diff, future_return, tau, target.
    """
    labeled = frame.copy()
    labeled = labeled.sort_values(by=[asset_column, date_column]).reset_index(drop=True)
    labeled[price_column] = pd.to_numeric(labeled[price_column], errors="coerce")

    grouped_prices = labeled.groupby(asset_column, sort=False)[price_column]
    labeled["future_price_diff"] = grouped_prices.shift(-horizon_days) - labeled[price_column]
    labeled["future_return"] = labeled["future_price_diff"] / labeled[price_column].replace(0.0, np.nan)

    grouped_returns = labeled.groupby(asset_column, sort=False)["future_return"]
    rolling_volatility = grouped_returns.transform(
        lambda series: series.rolling(
            window=rolling_window,
            min_periods=rolling_min_periods,
        ).std()
    )

    fallback_volatility = float(labeled["future_return"].std(skipna=True))
    if np.isnan(fallback_volatility):
        fallback_volatility = 0.0

    labeled["tau"] = rolling_volatility.fillna(fallback_volatility).abs() * vol_multiplier

    labeled[target_column] = 0
    labeled.loc[labeled["future_return"] > labeled["tau"], target_column] = 1
    labeled.loc[labeled["future_return"] < -labeled["tau"], target_column] = -1

    return labeled


def build_labeled_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load consolidated data, generate dynamic-threshold labels, and persist output.

    Args:
        config: Full project configuration dictionary.

    Returns:
        Labeled dataframe.
    """
    paths_cfg = config["paths"]
    dataset_cfg = config["dataset"]
    labels_cfg = config["labels"]
    features_cfg = config["features"]

    input_path = Path(paths_cfg["consolidated_data"])
    output_path = Path(paths_cfg["labeled_data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    labeled = generate_targets_with_dynamic_threshold(
        frame=frame,
        asset_column=dataset_cfg["normalized_asset_column"],
        date_column=dataset_cfg["normalized_date_column"],
        price_column=dataset_cfg["price_column"],
        target_column=features_cfg["target_column"],
        horizon_days=labels_cfg["horizon_days"],
        rolling_window=labels_cfg["rolling_window"],
        rolling_min_periods=labels_cfg["rolling_min_periods"],
        vol_multiplier=labels_cfg["vol_multiplier"],
    )

    labeled.to_parquet(output_path, index=False)
    LOGGER.info("Saved labeled dataset with %d rows -> %s", len(labeled), output_path)
    return labeled


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Generate CLEF labels from consolidated data.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """Run label generation as a standalone script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    build_labeled_dataset(config)


if __name__ == "__main__":
    main()
