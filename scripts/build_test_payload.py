from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_post_payload import (
    _coerce_optional_text,
    _format_date,
    _normalize_momentum,
    _resolve_column,
    load_config,
    load_source_frame,
)
from src.api.main import InferenceService, PredictRequest


def _prepare_frame(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Prepare and normalize source frame for candidate search."""
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

    momentum_candidates = [
        dataset_cfg.get("momentum_column", "momentum"),
        dataset_cfg.get("normalized_momentum_column", "momentum_numeric"),
    ]
    news_candidates = [
        dataset_cfg.get("news_column", "news"),
        dataset_cfg.get("normalized_text_column", "news_combined"),
    ]
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
    prepared[asset_col] = prepared[asset_col].astype(str).str.upper().str.strip()
    prepared = prepared.sort_values(by=[date_col, asset_col], ascending=[True, True]).reset_index(drop=True)

    if prepared.empty:
        raise ValueError("Source dataframe has no valid rows with date and price.")

    columns = {
        "asset": asset_col,
        "date": date_col,
        "price": price_col,
        "momentum": momentum_col,
        "news": news_col,
        "filing_10k": filing_10k_col,
        "filing_10q": filing_10q_col,
    }
    return prepared, columns


def _build_payload_from_row(
    prepared: pd.DataFrame,
    row_index: int,
    columns: dict[str, str],
    config: dict[str, Any],
    history_window: int,
) -> dict[str, Any]:
    """Build nested API payload for one specific row index."""
    asset_col = columns["asset"]
    date_col = columns["date"]
    price_col = columns["price"]
    momentum_col = columns["momentum"]
    news_col = columns["news"]
    filing_10k_col = columns["filing_10k"]
    filing_10q_col = columns["filing_10q"]

    row = prepared.iloc[row_index]
    asset = str(row[asset_col]).strip().upper()
    latest_date = pd.to_datetime(row[date_col], errors="coerce")
    latest_price = float(row[price_col])

    asset_history = prepared[prepared[asset_col] == asset]
    prior_rows = asset_history[asset_history[date_col] < latest_date].tail(history_window)
    history_payload = [
        {"date": _format_date(item[date_col]), "price": float(item[price_col])}
        for _, item in prior_rows.iterrows()
    ]

    news_text = _coerce_optional_text(row[news_col])
    filing_10k_text = _coerce_optional_text(row[filing_10k_col]) if filing_10k_col else None
    filing_10q_text = _coerce_optional_text(row[filing_10q_col]) if filing_10q_col else None

    payload = {
        "date": _format_date(latest_date),
        "price": {asset: latest_price},
        "symbol": {asset: asset},
        "momentum": {asset: _normalize_momentum(row[momentum_col], config)},
        "news": {asset: [news_text] if news_text else []},
        "10k": {asset: filing_10k_text},
        "10q": {asset: filing_10q_text},
        "history_price": {asset: history_payload},
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a target-decision API payload (BUY by default) from recent datapoints."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("test.json"))
    parser.add_argument("--target", type=str, default="BUY", choices=["BUY", "HOLD", "SELL"])
    parser.add_argument("--asset", type=str, default=None, help="Optional asset filter, e.g. BTC")
    parser.add_argument(
        "--history-window",
        type=int,
        default=None,
        help="How many prior prices to include in history_price",
    )
    parser.add_argument(
        "--max-per-asset",
        type=int,
        default=40,
        help="Max recent rows to probe per asset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frame = load_source_frame(config, args.source)
    prepared, columns = _prepare_frame(frame, config)

    default_window = int(config.get("features", {}).get("api_price_history_window", 60))
    history_window = args.history_window if args.history_window is not None else default_window

    service = InferenceService(args.config)
    target = args.target.upper()

    asset_values = sorted(prepared[columns["asset"]].dropna().astype(str).str.upper().unique())
    if args.asset:
        requested = args.asset.strip().upper()
        asset_values = [asset for asset in asset_values if asset == requested]
        if not asset_values:
            raise ValueError(f"Asset {requested} not found in source data")

    attempts = 0
    for asset in asset_values:
        asset_rows = prepared[prepared[columns["asset"]] == asset]
        asset_rows = asset_rows.sort_values(columns["date"], ascending=False).head(args.max_per_asset)

        for row_index in asset_rows.index:
            payload = _build_payload_from_row(
                prepared=prepared,
                row_index=int(row_index),
                columns=columns,
                config=config,
                history_window=history_window,
            )
            request_obj = PredictRequest(**payload)
            decision, _rationale = service.predict(request_obj)
            attempts += 1

            if decision == target:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w", encoding="utf-8") as file_obj:
                    json.dump(payload, file_obj, indent=2)
                    file_obj.write("\n")
                print(
                    f"Saved {args.output} with decision={decision} asset={asset} "
                    f"date={payload['date']} attempts={attempts}"
                )
                return

    raise RuntimeError(
        f"No payload produced decision={target} after {attempts} attempts. "
        "Try increasing --max-per-asset or removing --asset filter."
    )


if __name__ == "__main__":
    main()