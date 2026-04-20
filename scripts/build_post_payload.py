from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


# Load YAML config file.
def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


# Normalize optional text-like values into a single string or None.
def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, list):
        text = " ".join(str(item) for item in value if item is not None)
        text = " ".join(text.split())
        return text if text else None

    if isinstance(value, tuple):
        text = " ".join(str(item) for item in value if item is not None)
        text = " ".join(text.split())
        return text if text else None

    if isinstance(value, np.ndarray):
        text = " ".join(str(item) for item in value.tolist() if item is not None)
        text = " ".join(text.split())
        return text if text else None

    if isinstance(value, dict):
        text = " ".join(str(item) for item in value.values() if item is not None)
        text = " ".join(text.split())
        return text if text else None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            try:
                unwrapped = json.loads(stripped)
                if isinstance(unwrapped, str):
                    stripped = unwrapped.strip()
            except Exception:
                pass

        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("{") and stripped.endswith("}")
        ):
            try:
                parsed = json.loads(stripped)
                return _coerce_optional_text(parsed)
            except Exception:
                try:
                    parsed = ast.literal_eval(stripped)
                    return _coerce_optional_text(parsed)
                except Exception:
                    try:
                        repaired = stripped.replace('\\"', '"')
                        parsed = json.loads(repaired)
                        return _coerce_optional_text(parsed)
                    except Exception:
                        pass
        text = " ".join(stripped.split())
        return text if text else None

    text = " ".join(str(value).split())
    return text if text else None


# Format datetime-like value to YYYY-MM-DD.
def _format_date(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Unable to parse date value: {value}")
    return timestamp.strftime("%Y-%m-%d")


# Choose input parquet path.
def _pick_source_path(config: dict[str, Any], source_override: Optional[Path]) -> Path:
    if source_override is not None:
        return source_override

    paths_cfg = config["paths"]
    consolidated = Path(paths_cfg["consolidated_data"])
    if consolidated.exists():
        return consolidated

    labeled = Path(paths_cfg.get("labeled_data", ""))
    if labeled.exists():
        return labeled

    raise FileNotFoundError("No local parquet source found.")


# Resolve HF token from config or environment variable.
def _resolve_hf_token(dataset_cfg: dict[str, Any]) -> Optional[str]:
    configured_token = str(dataset_cfg.get("hf_token", "")).strip()
    if configured_token:
        return configured_token

    token_env_var = str(dataset_cfg.get("hf_token_env_var", "HF_TOKEN")).strip() or "HF_TOKEN"
    env_token = os.getenv(token_env_var, "").strip()
    return env_token or None


# Load and combine asset frames from HF parquet locations.
def _load_frame_from_hf(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    assets = [str(asset).upper() for asset in config["assets"]["tickers"]]

    repo_id = str(dataset_cfg.get("hf_parquet_repo", dataset_cfg.get("hf_dataset_name", ""))).strip()
    if not repo_id:
        raise FileNotFoundError("HF parquet repo is not configured and no local source exists.")

    configured_paths = {
        str(asset_key).strip().upper(): str(path_value).strip().lstrip("/")
        for asset_key, path_value in dict(dataset_cfg.get("hf_parquet_paths", {})).items()
        if str(path_value).strip()
    }

    if not configured_paths:
        configured_paths = {
            asset: f"data/{asset}-00000-of-00001.parquet"
            for asset in assets
        }

    token = _resolve_hf_token(dataset_cfg)
    storage_options = {"token": token} if token else {}

    combined_frames: list[pd.DataFrame] = []
    for asset in assets:
        relative_path = configured_paths.get(asset)
        if not relative_path:
            continue

        uri = f"hf://datasets/{repo_id}/{relative_path}"
        try:
            if storage_options:
                frame = pd.read_parquet(uri, storage_options=storage_options)
            else:
                frame = pd.read_parquet(uri)

            if "asset" not in frame.columns and "symbol" not in frame.columns and "ticker" not in frame.columns:
                frame = frame.copy()
                frame[dataset_cfg.get("normalized_asset_column", "asset")] = asset

            combined_frames.append(frame)
        except Exception:
            continue

    if not combined_frames:
        raise FileNotFoundError(
            "Unable to load any local or HF parquet sources. "
            "Run data fetching first or provide --source to an existing parquet."
        )

    return pd.concat(combined_frames, axis=0, ignore_index=True)


# Load source dataframe from local or HF parquet.
def load_source_frame(config: dict[str, Any], source_override: Optional[Path]) -> pd.DataFrame:
    try:
        source_path = _pick_source_path(config, source_override)
        return pd.read_parquet(source_path)
    except FileNotFoundError:
        return _load_frame_from_hf(config)


# Return the first existing column from candidates.
def _resolve_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of the candidate columns exist: {candidates}")


# Keep momentum as string when possible.
def _normalize_momentum(value: Any, config: dict[str, Any]) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "missing"

    if isinstance(value, str):
        text = value.strip().lower()
        return text if text else "missing"

    if isinstance(value, (int, float)):
        mapping = config["dataset"].get("momentum_mapping", {})
        reverse_map = {int(v): k for k, v in mapping.items()}
        numeric = int(value)
        return reverse_map.get(numeric, numeric)

    return str(value)


# Create nested API payload from the latest datapoint.
def build_payload(
    frame: pd.DataFrame,
    config: dict[str, Any],
    asset_override: Optional[str],
    history_window: int,
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]

    asset_col = _resolve_column(
        frame,
        [dataset_cfg.get("normalized_asset_column", "asset")]
        + dataset_cfg.get("asset_column_candidates", []),
    )
    date_col = _resolve_column(
        frame,
        [dataset_cfg.get("normalized_date_column", "date")]
        + dataset_cfg.get("date_column_candidates", []),
    )
    price_col = _resolve_column(frame, [dataset_cfg.get("price_column", "prices")])

    momentum_candidates = [dataset_cfg.get("momentum_column", "momentum"), dataset_cfg.get("normalized_momentum_column", "momentum_numeric")]
    news_candidates = [dataset_cfg.get("news_column", "news"), dataset_cfg.get("normalized_text_column", "news_combined")]
    filing_10k_candidates = [dataset_cfg.get("tenk_column", "filing_10k")]
    filing_10q_candidates = [dataset_cfg.get("tenq_column", "filing_10q")]

    momentum_col = _resolve_column(frame, [candidate for candidate in momentum_candidates if candidate])
    news_col = _resolve_column(frame, [candidate for candidate in news_candidates if candidate])

    filing_10k_col = None
    for candidate in filing_10k_candidates:
        if candidate and candidate in frame.columns:
            filing_10k_col = candidate
            break

    filing_10q_col = None
    for candidate in filing_10q_candidates:
        if candidate and candidate in frame.columns:
            filing_10q_col = candidate
            break

    prepared = frame.copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
    prepared = prepared.dropna(subset=[date_col, price_col])
    prepared = prepared.sort_values(by=[date_col, asset_col], ascending=[True, True])

    if prepared.empty:
        raise ValueError("Source dataframe has no valid rows with date and price.")

    if asset_override:
        target_asset = asset_override.strip().upper()
        prepared_asset = prepared[prepared[asset_col].astype(str).str.upper() == target_asset]
        if prepared_asset.empty:
            raise ValueError(f"No rows found for asset={target_asset}")
        latest_row = prepared_asset.iloc[-1]
    else:
        latest_row = prepared.iloc[-1]

    asset = str(latest_row[asset_col]).strip().upper()
    latest_date = pd.to_datetime(latest_row[date_col], errors="coerce")
    latest_price = float(latest_row[price_col])

    asset_history = prepared[prepared[asset_col].astype(str).str.upper() == asset].copy()
    prior_rows = asset_history[asset_history[date_col] < latest_date].tail(history_window)

    history_payload = [
        {
            "date": _format_date(row[date_col]),
            "price": float(row[price_col]),
        }
        for _, row in prior_rows.iterrows()
    ]

    news_text = _coerce_optional_text(latest_row[news_col])
    filing_10k_text = _coerce_optional_text(latest_row[filing_10k_col]) if filing_10k_col else None
    filing_10q_text = _coerce_optional_text(latest_row[filing_10q_col]) if filing_10q_col else None

    payload = {
        "date": _format_date(latest_date),
        "price": {asset: latest_price},
        "symbol": {asset: asset},
        "momentum": {asset: _normalize_momentum(latest_row[momentum_col], config)},
        "news": {asset: [news_text] if news_text else []},
        "10k": {asset: filing_10k_text},
        "10q": {asset: filing_10q_text},
        "history_price": {asset: history_payload},
    }

    return payload


# CLI argument parser.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a nested FastAPI /predict payload from the latest datapoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional override parquet source (default: consolidated_data from config).",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Optional asset override (e.g., BTC). If omitted, uses latest row overall.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=None,
        help="How many prior prices to include in history_price.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("post.json"),
        help="Output JSON file path.",
    )
    return parser.parse_args()


# Read latest datapoint, convert to API payload, and write JSON.
def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    frame = load_source_frame(config, args.source)

    default_window = int(config.get("features", {}).get("api_price_history_window", 60))
    history_window = args.history_window if args.history_window is not None else default_window

    payload = build_payload(
        frame=frame,
        config=config,
        asset_override=args.asset,
        history_window=history_window,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
        file_obj.write("\n")

    print(f"Saved payload to {args.output}")


if __name__ == "__main__":
    main()
