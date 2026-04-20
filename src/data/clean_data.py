from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _resolve_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    existing_columns = set(frame.columns)
    for candidate in candidates:
        if candidate in existing_columns:
            return candidate
    return None

# COnvert optional array-like filing fields into a single plain string.
def _coerce_optional_array_text(value: Any) -> str:
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

# Normalize whitespace and missing values for text fields.
def _normalize_text(series: pd.Series) -> pd.Series:
    
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

# Return an existing column or a default-filled series with aligned index.
def _safe_series(frame: pd.DataFrame, column_name: str, default: Any = "") -> pd.Series:
    
    if column_name in frame.columns:
        return frame[column_name]
    return pd.Series([default] * len(frame), index=frame.index)

# Map raw momentum values into normalized integer states based on config mapping.
def _map_momentum(value: Any, momentum_mapping: dict[str, int]) -> int:
    
    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return momentum_mapping.get("missing", 0)
        return int(value)

    if value is None:
        return momentum_mapping.get("missing", 0)

    mapped = momentum_mapping.get(str(value).strip().lower())
    if mapped is None:
        return momentum_mapping.get("missing", 0)
    return mapped

# Normalize a single asset dataframe into a consistent schema.
def _prepare_asset_frame(frame: pd.DataFrame, asset: str, config: dict[str, Any]) -> pd.DataFrame:
    
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

    cleaned[price_column] = pd.to_numeric(_safe_series(cleaned, price_column, np.nan), errors="coerce")
    cleaned[normalized_momentum_column] = _safe_series(cleaned, momentum_column, np.nan).apply(
        lambda value: _map_momentum(value, momentum_mapping)
    )

    base_news = _normalize_text(_safe_series(cleaned, news_column, ""))
    tenk_text = _safe_series(cleaned, tenk_column, "").apply(_coerce_optional_array_text)
    tenq_text = _safe_series(cleaned, tenq_column, "").apply(_coerce_optional_array_text)

    tenk_text = _normalize_text(tenk_text)
    tenq_text = _normalize_text(tenq_text)

    cleaned[normalized_text_column] = _normalize_text(base_news + " " + tenk_text + " " + tenq_text)

    return cleaned

# Load consolidated data, generate dynamic-threshold labels, and persist output.
def build_consolidated_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    
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
    parser = argparse.ArgumentParser(description="Clean and consolidate raw CLEF asset data.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    build_consolidated_dataframe(config)


if __name__ == "__main__":
    main()
