from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset

LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Absolute or relative path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _resolve_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Resolve the first existing column name from a candidate list.

    Args:
        frame: Input dataframe to inspect.
        candidates: Ordered list of potential column names.

    Returns:
        The first matching column name, or None when no match exists.
    """
    existing_columns = set(frame.columns)
    for candidate in candidates:
        if candidate in existing_columns:
            return candidate
    return None


def _resolve_hf_token(dataset_cfg: dict[str, Any]) -> Optional[str]:
    """Resolve Hugging Face auth token from config literal or environment variable.

    Args:
        dataset_cfg: Dataset section of project configuration.

    Returns:
        Token string when available, else None.
    """
    configured_token = str(dataset_cfg.get("hf_token", "")).strip()
    if configured_token:
        return configured_token

    token_env_var = _sanitize_env_var_name(str(dataset_cfg.get("hf_token_env_var", "HF_TOKEN")).strip())
    if not token_env_var:
        return None

    env_token = os.getenv(token_env_var, "").strip()
    if env_token:
        return env_token

    return None


def _sanitize_env_var_name(candidate: str) -> str:
    """Return a safe environment variable name or a default fallback.

    Args:
        candidate: Raw configured environment variable key.

    Returns:
        Sanitized env var name suitable for user-facing hints.
    """
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
        return candidate
    return "HF_TOKEN"


def _load_hf_dataset(
    dataset_name: str,
    split_name: str,
    token: Optional[str],
    subset_name: Optional[str] = None,
) -> Any:
    """Load Hugging Face dataset with token compatibility across library versions.

    Args:
        dataset_name: Dataset repository identifier.
        split_name: Split name to load.
        token: Optional authentication token.
        subset_name: Optional subset/configuration name.

    Returns:
        Loaded dataset split object.
    """
    kwargs: dict[str, Any] = {"split": split_name}
    if token:
        kwargs["token"] = token

    try:
        if subset_name is None:
            return load_dataset(dataset_name, **kwargs)
        return load_dataset(dataset_name, subset_name, **kwargs)
    except TypeError:
        # Backward-compatible path for older datasets versions.
        legacy_kwargs = {"split": split_name}
        if token:
            legacy_kwargs["use_auth_token"] = token
        if subset_name is None:
            return load_dataset(dataset_name, **legacy_kwargs)
        return load_dataset(dataset_name, subset_name, **legacy_kwargs)


def _create_synthetic_asset_frame(
    asset: str,
    rows_per_asset: int,
    start_date: str,
    dataset_cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Create synthetic asset data when remote CLEF data is unavailable.

    Args:
        asset: Asset ticker symbol.
        rows_per_asset: Number of synthetic daily rows to generate.
        start_date: Start date for generated daily timeline.
        dataset_cfg: Dataset configuration dictionary.
        seed: Base random seed for reproducibility.

    Returns:
        Synthetic dataframe following expected schema fields.
    """
    asset_seed = seed + sum(ord(char) for char in asset)
    rng = np.random.default_rng(asset_seed)

    dates = pd.date_range(start=start_date, periods=rows_per_asset, freq="D")
    daily_noise = rng.normal(loc=0.0, scale=0.012, size=rows_per_asset)

    base_price = float(100 + rng.uniform(10, 200))
    prices = np.empty(rows_per_asset, dtype=np.float64)
    prices[0] = base_price
    for idx in range(1, rows_per_asset):
        prices[idx] = max(1.0, prices[idx - 1] * (1.0 + daily_noise[idx]))

    momentum_raw = np.where(daily_noise > 0.003, "bullish", np.where(daily_noise < -0.003, "bearish", "missing"))

    news_column = dataset_cfg["news_column"]
    price_column = dataset_cfg["price_column"]
    momentum_column = dataset_cfg["momentum_column"]
    tenk_column = dataset_cfg["tenk_column"]
    tenq_column = dataset_cfg["tenq_column"]
    normalized_asset_column = dataset_cfg["normalized_asset_column"]

    frame = pd.DataFrame(
        {
            "date": dates,
            normalized_asset_column: asset,
            price_column: prices,
            momentum_column: momentum_raw,
            news_column: [
                f"Synthetic {asset} market update: volatility regime sample {i}."
                for i in range(rows_per_asset)
            ],
            tenk_column: "",
            tenq_column: "",
        }
    )
    return frame


def _build_synthetic_raw_files(config: dict[str, Any], raw_dir: Path) -> dict[str, Path]:
    """Generate synthetic raw parquet files for all configured assets.

    Args:
        config: Full project configuration.
        raw_dir: Output directory for raw parquet files.

    Returns:
        Mapping from asset ticker to generated parquet path.
    """
    dataset_cfg = config["dataset"]
    assets = config["assets"]["tickers"]
    rows_per_asset = int(dataset_cfg.get("synthetic_rows_per_asset", 120))
    start_date = str(dataset_cfg.get("synthetic_start_date", "2021-01-01"))
    seed = int(config.get("seed", {}).get("numpy", 42))

    output_paths: dict[str, Path] = {}
    for asset in assets:
        synthetic_frame = _create_synthetic_asset_frame(
            asset=asset,
            rows_per_asset=rows_per_asset,
            start_date=start_date,
            dataset_cfg=dataset_cfg,
            seed=seed,
        )
        output_path = raw_dir / f"{asset.lower()}.parquet"
        synthetic_frame.to_parquet(output_path, index=False)
        output_paths[asset] = output_path
        LOGGER.info("Saved synthetic %s rows for %s -> %s", len(synthetic_frame), asset, output_path)

    return output_paths


def fetch_assets(config: dict[str, Any]) -> dict[str, Path]:
    """Fetch asset-level CLEF records from Hugging Face and persist per-asset parquet files.

    This function supports both unified datasets (all assets in one split) and datasets
    where each asset is exposed as a subset configuration.

    Args:
        config: Full project configuration.

    Returns:
        Mapping from asset ticker to saved parquet path.

    Raises:
        RuntimeError: If no assets could be downloaded.
    """
    dataset_cfg = config["dataset"]
    paths_cfg = config["paths"]
    assets = config["assets"]["tickers"]

    dataset_name = dataset_cfg["hf_dataset_name"]
    split_name = dataset_cfg["hf_split"]
    asset_column_candidates = dataset_cfg["asset_column_candidates"]
    normalized_asset_column = dataset_cfg["normalized_asset_column"]
    subset_map = dataset_cfg.get("hf_asset_subsets", {})
    use_auth = bool(dataset_cfg.get("hf_use_auth_token", True))
    hf_token = _resolve_hf_token(dataset_cfg) if use_auth else None
    token_env_var = _sanitize_env_var_name(str(dataset_cfg.get("hf_token_env_var", "HF_TOKEN")).strip())

    if hf_token:
        LOGGER.info("Hugging Face token detected; authenticated dataset access enabled.")
    else:
        LOGGER.info("No Hugging Face token detected; attempting public dataset access.")

    raw_dir = Path(paths_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    shared_frame: Optional[pd.DataFrame] = None
    try:
        LOGGER.info("Loading shared dataset '%s' split '%s'", dataset_name, split_name)
        shared_dataset = _load_hf_dataset(
            dataset_name=dataset_name,
            split_name=split_name,
            token=hf_token,
        )
        shared_frame = shared_dataset.to_pandas()
        LOGGER.info("Loaded shared dataset with %d rows", len(shared_frame))
    except Exception as error:  # pragma: no cover - network and dataset variability
        LOGGER.warning("Shared dataset load failed, falling back to subset loads: %s", error)

    output_paths: dict[str, Path] = {}
    for asset in assets:
        asset_frame = pd.DataFrame()

        if shared_frame is not None and not shared_frame.empty:
            asset_column = _resolve_first_column(shared_frame, asset_column_candidates)
            if asset_column is not None:
                mask = shared_frame[asset_column].astype(str).str.upper() == asset.upper()
                asset_frame = shared_frame.loc[mask].copy()

        if asset_frame.empty:
            subset_name = subset_map.get(asset, asset)
            try:
                LOGGER.info(
                    "Loading asset subset '%s' for asset '%s' from '%s'",
                    subset_name,
                    asset,
                    dataset_name,
                )
                subset_dataset = _load_hf_dataset(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    token=hf_token,
                    subset_name=subset_name,
                )
                asset_frame = subset_dataset.to_pandas()
            except Exception as error:  # pragma: no cover - network and dataset variability
                LOGGER.error("Failed to fetch asset '%s': %s", asset, error)
                continue

        if normalized_asset_column not in asset_frame.columns:
            asset_frame[normalized_asset_column] = asset

        output_path = raw_dir / f"{asset.lower()}.parquet"
        asset_frame.to_parquet(output_path, index=False)
        output_paths[asset] = output_path
        LOGGER.info("Saved %d rows for %s -> %s", len(asset_frame), asset, output_path)

    if not output_paths:
        if bool(dataset_cfg.get("allow_synthetic_fallback", False)):
            LOGGER.warning(
                "No Hugging Face data could be fetched; generating synthetic fallback raw files."
            )
            return _build_synthetic_raw_files(config=config, raw_dir=raw_dir)

        auth_hint = (
            f" If this dataset is private, export {token_env_var}=<your_token> "
            "or set dataset.hf_token in config."
        )
        raise RuntimeError(
            "No asset files were created. Check dataset configuration and access rights."
            + auth_hint
        )

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Download CLEF asset files from Hugging Face.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """Run the asset downloader as a standalone script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    fetch_assets(config)


if __name__ == "__main__":
    main()
