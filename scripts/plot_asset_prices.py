from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def resolve_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first existing column from ordered candidates."""
    existing = set(frame.columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def plot_asset_price(
    parquet_path: Path,
    output_dir: Path,
    date_candidates: list[str],
    price_column: str,
) -> Optional[Path]:
    """Create and save one price-vs-date plot for a single asset parquet file."""
    asset = parquet_path.stem.upper()
    frame = pd.read_parquet(parquet_path)

    date_column = resolve_first_column(frame, date_candidates)
    if date_column is None:
        LOGGER.warning("Skipping %s because no date column was found.", parquet_path.name)
        return None

    if price_column not in frame.columns:
        LOGGER.warning(
            "Skipping %s because price column '%s' was not found.",
            parquet_path.name,
            price_column,
        )
        return None

    prepared = frame[[date_column, price_column]].copy()
    prepared[date_column] = pd.to_datetime(prepared[date_column], errors="coerce", utc=True)
    prepared[date_column] = prepared[date_column].dt.tz_localize(None)
    prepared[price_column] = pd.to_numeric(prepared[price_column], errors="coerce")
    prepared = prepared.dropna(subset=[date_column, price_column]).sort_values(by=date_column)

    if prepared.empty:
        LOGGER.warning("Skipping %s because no valid rows remained after cleanup.", parquet_path.name)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{asset}_price_vs_date.png"

    plt.figure(figsize=(12, 5))
    plt.plot(prepared[date_column], prepared[price_column], linewidth=1.8)
    plt.title(f"{asset} Price vs Date")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    LOGGER.info("Saved plot for %s -> %s", asset, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot price vs date for each asset parquet file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to project config yaml.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional override for raw parquet directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and save one plot per asset parquet in the input directory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})

    raw_dir = args.input_dir or Path(paths_cfg.get("raw_dir", "data/raw"))
    price_column = str(dataset_cfg.get("price_column", "prices"))
    date_candidates = [
        str(dataset_cfg.get("normalized_date_column", "date")),
        *[str(item) for item in dataset_cfg.get("date_column_candidates", ["date", "datetime", "timestamp"])],
    ]

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    saved_paths: list[Path] = []
    for parquet_path in parquet_files:
        output_path = plot_asset_price(
            parquet_path=parquet_path,
            output_dir=args.output_dir,
            date_candidates=date_candidates,
            price_column=price_column,
        )
        if output_path is not None:
            saved_paths.append(output_path)

    LOGGER.info("Completed plotting. Saved %d file(s) to %s", len(saved_paths), args.output_dir)


if __name__ == "__main__":
    main()
