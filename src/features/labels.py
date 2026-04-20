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
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


# Create return labels using a rolling volatility-dependent threshold.
def generate_targets_with_dynamic_threshold(
    frame: pd.DataFrame,
    asset_column: str,
    date_column: str,
    price_column: str,
    target_column: str,
    horizon_days: int,
    rolling_window: int,
    rolling_min_periods: int,
    vol_multiplier_buy: float,
    vol_multiplier_sell: float,
) -> pd.DataFrame:
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
        )
        .std()
        .shift(1)
    )

    fallback_volatility = float(labeled["future_return"].std(skipna=True))
    if np.isnan(fallback_volatility):
        fallback_volatility = 0.0

    rolling_volatility = rolling_volatility.fillna(fallback_volatility).abs()
    labeled["tau_buy"] = rolling_volatility * vol_multiplier_buy
    labeled["tau_sell"] = rolling_volatility * vol_multiplier_sell

    valid_future_mask = labeled["future_return"].notna()
    labeled[target_column] = np.nan
    labeled.loc[valid_future_mask, target_column] = 0
    labeled.loc[valid_future_mask & (labeled["future_return"] > labeled["tau_buy"]), target_column] = 1
    labeled.loc[valid_future_mask & (labeled["future_return"] < -labeled["tau_sell"]), target_column] = -1

    return labeled


# Load consolidated data, generate dynamic-threshold labels, and persist output.
def build_labeled_dataset(config: dict[str, Any]) -> pd.DataFrame:
    paths_cfg = config["paths"]
    dataset_cfg = config["dataset"]
    labels_cfg = config["labels"]
    features_cfg = config["features"]

    input_path = Path(paths_cfg["consolidated_data"])
    output_path = Path(paths_cfg["labeled_data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    labeled_frame = generate_targets_with_dynamic_threshold(
        frame=frame,
        asset_column=dataset_cfg["normalized_asset_column"],
        date_column=dataset_cfg["normalized_date_column"],
        price_column=dataset_cfg["price_column"],
        target_column=features_cfg["target_column"],
        horizon_days=int(labels_cfg["horizon_days"]),
        rolling_window=int(labels_cfg["rolling_window"]),
        rolling_min_periods=int(labels_cfg["rolling_min_periods"]),
        vol_multiplier_buy=float(labels_cfg["vol_multiplier_buy"]),
        vol_multiplier_sell=float(labels_cfg["vol_multiplier_sell"]),
    )

    labeled_frame.to_parquet(output_path, index=False)
    LOGGER.info("Saved labeled dataset with %d rows -> %s", len(labeled_frame), output_path)
    return labeled_frame


# Parse command-line arguments for standalone execution.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLEF labels from consolidated data.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    build_labeled_dataset(config)


if __name__ == "__main__":
    main()
