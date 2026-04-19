from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from src.utils import load_config, _resolve_first_column

LOGGER = logging.getLogger(__name__)


def _coerce_optional_array_text(value: Any) -> str:
    """Convert optional array-like filing fields into a single plain string.

    This function exists to neutralize schema drift across CLEF asset files where
    10-K and 10-Q columns can be null, list-like, or scalar text values.

    Args:
        value: Raw field value from the dataframe.

    Returns:
        Normalized text string.
    """
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None)

    if isinstance(value, tuple):
        return " ".join(str(item) for item in value if item is not None)

    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values() if item is not None)

    return str(value)


def _normalize_text(series: pd.Series) -> pd.Series:
    """Normalize whitespace and missing values for text fields.

    Args:
        series: Input text-like series.

    Returns:
        Cleaned text series.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _prepare_asset_frame(frame: pd.DataFrame, asset: str, config: dict[str, Any]) -> pd.DataFrame:
    """Normalize a single asset dataframe into a consistent schema.

    Args:
        frame: Raw asset dataframe.
        asset: Asset ticker symbol.
        config: Full project configuration.

    Returns:
        Schema-normalized dataframe for one asset.

    Raises:
        ValueError: If a usable date column cannot be resolved.
    """
    dataset_cfg = config["dataset"]

    date_column = _resolve_first_column(frame, dataset_cfg["date_column_candidates"])
    if date_column is None:
        raise ValueError(f"No date column found for asset {asset}")

    asset_column = _resolve_first_column(frame, dataset_cfg["asset_column_candidates"])

    price_column = dataset_cfg["price_column"]
    momentum_column = dataset_cfg["momentum_column"]
    news_column = dataset_cfg["news_column"]
    tenk_column = dataset_cfg["tenk_column"]
    tenq_column = dataset_cfg["tenq_column"]

    normalized_asset_column = dataset_cfg["normalized_asset_column"]
    normalized_date_column = dataset_cfg["normalized_date_column"]
    normalized_momentum_column = dataset_cfg["normalized_momentum_column"]
    normalized_text_column = dataset_cfg["normalized_text_column"]

    momentum_mapping = dataset_cfg["momentum_mapping"]

    cleaned = frame.copy()
    if asset_column is not None:
        cleaned[normalized_asset_column] = cleaned[asset_column].fillna(asset).astype(str)
    else:
        cleaned[normalized_asset_column] = asset

    cleaned[normalized_date_column] = pd.to_datetime(cleaned[date_column], errors="coerce", utc=True)
    cleaned[normalized_date_column] = cleaned[normalized_date_column].dt.tz_localize(None)

    # News and Price are essential. If missing, this will fail or produce NaNs for filtering later.
    cleaned[price_column] = pd.to_numeric(cleaned[price_column], errors="coerce")
    
    # Simple momentum mapping
    def map_mom(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return momentum_mapping.get("missing", 0)
        m = momentum_mapping.get(str(val).strip().lower())
        return m if m is not None else momentum_mapping.get("missing", 0)

    cleaned[normalized_momentum_column] = cleaned[momentum_column].apply(map_mom)

    base_news = cleaned[news_column].fillna("").astype(str)
    tenk_text = (cleaned[tenk_column] if tenk_column in cleaned.columns else pd.Series([""] * len(cleaned))).apply(_coerce_optional_array_text)
    tenq_text = (cleaned[tenq_column] if tenq_column in cleaned.columns else pd.Series([""] * len(cleaned))).apply(_coerce_optional_array_text)

    # Perform one-pass normalization at the end
    cleaned[normalized_text_column] = _normalize_text(base_news + " " + tenk_text + " " + tenq_text)

    return cleaned


def build_consolidated_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    """Build and save a consolidated chronological dataset across all configured assets.

    Args:
        config: Full project configuration.

    Returns:
        Consolidated dataframe sorted by date and asset.

    Raises:
        RuntimeError: If no raw asset files are available.
    """
    paths_cfg = config["paths"]
    dataset_cfg = config["dataset"]

    raw_dir = Path(paths_cfg["raw_dir"])
    processed_output = Path(paths_cfg["consolidated_data"])
    processed_output.parent.mkdir(parents=True, exist_ok=True)

    assets = config["assets"]["tickers"]
    prepared_frames: list[pd.DataFrame] = []

    for asset in assets:
        input_path = raw_dir / f"{asset.lower()}.parquet"
        if not input_path.exists():
            LOGGER.warning("Skipping missing raw file: %s", input_path)
            continue

        raw_frame = pd.read_parquet(input_path)
        prepared_frame = _prepare_asset_frame(raw_frame, asset, config)
        prepared_frames.append(prepared_frame)

    if not prepared_frames:
        raise RuntimeError("No raw asset files found for consolidation.")

    consolidated = pd.concat(prepared_frames, axis=0, ignore_index=True)
    consolidated = consolidated.dropna(subset=[dataset_cfg["normalized_date_column"], dataset_cfg["price_column"]])
    consolidated = consolidated.sort_values(
        by=[dataset_cfg["normalized_date_column"], dataset_cfg["normalized_asset_column"]],
        ascending=[True, True],
    ).reset_index(drop=True)

    consolidated.to_parquet(processed_output, index=False)
    LOGGER.info("Saved consolidated dataset with %d rows -> %s", len(consolidated), processed_output)
    return consolidated


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Clean and consolidate raw CLEF asset data.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """Run cleaning and consolidation as a standalone script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    build_consolidated_dataframe(config)


if __name__ == "__main__":
    main()
