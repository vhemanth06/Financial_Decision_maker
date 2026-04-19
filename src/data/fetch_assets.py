from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from datasets import load_dataset

from src.utils import load_config, _resolve_first_column

LOGGER = logging.getLogger(__name__)


# def _resolve_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
#     """Resolve the first existing column name from a candidate list.

#     Args:
#         frame: Input dataframe to inspect.
#         candidates: Ordered list of potential column names.

#     Returns:
#         The first matching column name, or None when no match exists.
#     """
#     existing_columns = set(frame.columns)
#     for candidate in candidates:
#         if candidate in existing_columns:
#             return candidate
#     return None


def _resolve_hf_token(dataset_cfg: dict[str, Any]) -> Optional[str]:
    """Resolve Hugging Face auth token from config literal or environment variable.

    Args:
        dataset_cfg: Dataset section of project configuration.

    Returns:
        Token string when available, else None.
    """
    token_env_var = str(dataset_cfg.get("hf_token_env_var", "HF_TOKEN")).strip()

    configured_token = str(dataset_cfg.get("hf_token", "")).strip()
    if configured_token:
        # A simple check for a real-looking token.
        if len(configured_token) > 20 and "token" not in configured_token.lower():
            return configured_token

    env_token = os.getenv(token_env_var, "").strip()
    return env_token if env_token else None


def _load_hf_dataset(
    dataset_name: str,
    split_name: str,
    token: Optional[str],
    subset_name: Optional[str] = None,
) -> Any:
    """Load Hugging Face dataset split directly.

    Args:
        dataset_name: Dataset repository identifier.
        split_name: Split name to load.
        token: Optional authentication token.
        subset_name: Optional subset/configuration name.

    Returns:
        Loaded dataset split object.
    """
    kwargs: dict[str, Any] = {"split": split_name, "token": token}

    if subset_name is None:
        return load_dataset(dataset_name, **kwargs)
    return load_dataset(dataset_name, subset_name, **kwargs)


def _try_hf_login(token: Optional[str]) -> None:
    """Attempt an explicit Hugging Face login for hf:// parquet access.

    Args:
        token: Optional HF token.
    """
    if not token:
        return

    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        LOGGER.info("Authenticated with huggingface_hub.login")
    except Exception as error:  # pragma: no cover - external auth variability
        LOGGER.warning("huggingface_hub login failed; continuing with tokenized access: %s", error)


def _build_default_hf_parquet_paths(assets: list[str]) -> dict[str, str]:
    """Build default asset->relative parquet path mapping for hf:// reads.

    Args:
        assets: Configured asset tickers.

    Returns:
        Mapping from uppercase ticker to relative parquet path.
    """
    return {
        asset.upper(): f"data/{asset.upper()}-00000-of-00001.parquet"
        for asset in assets
    }


def _build_hf_storage_options(token: Optional[str]) -> dict[str, Any]:
    """Build storage options for fsspec hf:// access.

    Args:
        token: Optional authentication token.

    Returns:
        Storage options dictionary for pandas.read_parquet.
    """
    if not token:
        return {}
    return {"token": token}


def _load_hf_parquet_frame(uri: str, token: Optional[str]) -> pd.DataFrame:
    """Load one dataframe from hf:// parquet URI.

    Args:
        uri: Full hf:// parquet URI.
        token: Optional token used by storage_options.

    Returns:
        Loaded dataframe.
    """
    storage_options = _build_hf_storage_options(token)
    if storage_options:
        return pd.read_parquet(uri, storage_options=storage_options)
    return pd.read_parquet(uri)


def _load_asset_frames_from_hf_parquet(
    dataset_cfg: dict[str, Any],
    assets: list[str],
    token: Optional[str],
) -> dict[str, pd.DataFrame]:
    """Load per-asset frames using direct hf:// parquet paths.

    Args:
        dataset_cfg: Dataset configuration dictionary.
        assets: List of configured tickers.
        token: Optional token for authenticated dataset access.

    Returns:
        Mapping from ticker to dataframe for successfully loaded assets.
    """
    repo_id = str(dataset_cfg.get("hf_parquet_repo", dataset_cfg.get("hf_dataset_name", ""))).strip()
    if not repo_id:
        return {}

    configured_paths = dataset_cfg.get("hf_parquet_paths", {})
    if configured_paths:
        parquet_paths = {
            str(asset_key).strip().upper(): str(path_value).strip().lstrip("/")
            for asset_key, path_value in dict(configured_paths).items()
            if str(path_value).strip()
        }
    else:
        parquet_paths = _build_default_hf_parquet_paths(assets)

    loaded_frames: dict[str, pd.DataFrame] = {}
    for asset in assets:
        relative_path = parquet_paths.get(asset.upper())
        if not relative_path:
            continue

        uri = f"hf://datasets/{repo_id}/{relative_path}"
        try:
            frame = _load_hf_parquet_frame(uri=uri, token=token)
            loaded_frames[asset] = frame
            LOGGER.info("Loaded %d rows for %s from %s", len(frame), asset, uri)
        except Exception as error:  # pragma: no cover - network and dataset variability
            LOGGER.error("Failed hf:// parquet read for %s from %s: %s", asset, uri, error)

    return loaded_frames


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
    split_name = str(dataset_cfg.get("hf_split", "")).strip()
    asset_column_candidates = dataset_cfg["asset_column_candidates"]
    normalized_asset_column = dataset_cfg["normalized_asset_column"]
    subset_map = dataset_cfg.get("hf_asset_subsets", {})
    use_auth = bool(dataset_cfg.get("hf_use_auth_token", True))
    hf_token = _resolve_hf_token(dataset_cfg) if use_auth else None
    token_env_var = str(dataset_cfg.get("hf_token_env_var", "HF_TOKEN")).strip()

    if hf_token:
        LOGGER.info("Hugging Face token detected; authenticated dataset access enabled.")
    else:
        LOGGER.info("No Hugging Face token detected; attempting public dataset access.")

    _try_hf_login(hf_token)

    raw_dir = Path(paths_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Primary loading path: Try direct hf:// parquet reads for each asset if repo is specified.
    parquet_frames = _load_asset_frames_from_hf_parquet(
        dataset_cfg=dataset_cfg,
        assets=assets,
        token=hf_token,
    )

    output_paths: dict[str, Path] = {}
    for asset in assets:
        asset_frame = pd.DataFrame()

        # Step 1: Use direct parquet data if available
        if asset in parquet_frames:
            asset_frame = parquet_frames[asset].copy()

        # Step 2: Fallback to subset-level load_dataset if parquet didn't provide it
        if asset_frame.empty:
            subset_name = subset_map.get(asset, asset)
            try:
                LOGGER.info("Attempting asset subset load for '%s' from '%s'", subset_name, dataset_name)
                subset_dataset = _load_hf_dataset(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    token=hf_token,
                    subset_name=subset_name,
                )
                asset_frame = subset_dataset.to_pandas()
            except Exception as error:
                LOGGER.error("Failed to fetch asset '%s': %s", asset, error)

        if asset_frame.empty:
            continue

        if normalized_asset_column not in asset_frame.columns:
            asset_frame[normalized_asset_column] = asset

        output_path = raw_dir / f"{asset.lower()}.parquet"
        asset_frame.to_parquet(output_path, index=False)
        output_paths[asset] = output_path
        LOGGER.info("Saved %d rows for %s -> %s", len(asset_frame), asset, output_path)

    if not output_paths:
        auth_hint = f" If this dataset is private, export {token_env_var}=<your_token>."
        raise RuntimeError("No asset files were created. Check access rights." + auth_hint)

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
