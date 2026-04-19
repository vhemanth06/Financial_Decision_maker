from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

LABEL_TO_DECISION = {1: "BUY", 0: "HOLD", -1: "SELL"}
PLOT_ORDER = ["BUY", "HOLD", "SELL"]
PLOT_COLORS = ["#2e7d32", "#6d6d6d", "#c62828"]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _prepare_training_labels(config: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    """Load labeled training data and normalize target labels for plotting."""
    paths_cfg = config["paths"]
    dataset_cfg = config["dataset"]
    features_cfg = config["features"]

    labeled_path = Path(paths_cfg["labeled_data"])
    asset_col = str(dataset_cfg.get("normalized_asset_column", "asset"))
    target_col = str(features_cfg.get("target_column", "target"))

    frame = pd.read_parquet(labeled_path)
    if asset_col not in frame.columns:
        raise KeyError(f"Asset column '{asset_col}' not found in {labeled_path}")
    if target_col not in frame.columns:
        raise KeyError(f"Target column '{target_col}' not found in {labeled_path}")

    prepared = frame[[asset_col, target_col]].copy()
    prepared[asset_col] = prepared[asset_col].astype(str).str.upper().str.strip()
    prepared[target_col] = pd.to_numeric(prepared[target_col], errors="coerce")
    prepared = prepared[prepared[target_col].isin(LABEL_TO_DECISION.keys())]
    prepared[target_col] = prepared[target_col].astype(int)
    prepared["decision"] = prepared[target_col].map(LABEL_TO_DECISION)

    return prepared, asset_col, target_col


def _build_counts_for_asset(prepared: pd.DataFrame, asset_col: str, asset: str) -> pd.Series:
    """Return BUY/HOLD/SELL count series for one asset in stable plot order."""
    counts = (
        prepared.loc[prepared[asset_col] == asset, "decision"]
        .value_counts()
        .reindex(PLOT_ORDER, fill_value=0)
    )
    return counts


def _plot_counts(asset: str, counts: pd.Series, output_path: Path) -> None:
    """Plot and save one bar chart for a single asset distribution."""
    total = int(counts.sum())
    percentages = counts / total * 100 if total > 0 else counts

    plt.figure(figsize=(8, 5))
    bars = plt.bar(PLOT_ORDER, counts.values, color=PLOT_COLORS)
    plt.title(f"{asset} Training Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_action_distributions(config_path: Path, output_dir: Path) -> list[Path]:
    """Generate per-asset BUY/HOLD/SELL distribution plots for training data."""
    config = load_config(config_path)
    prepared, asset_col, _target_col = _prepare_training_labels(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    assets = sorted(prepared[asset_col].dropna().unique().tolist())
    for asset in assets:
        counts = _build_counts_for_asset(prepared=prepared, asset_col=asset_col, asset=asset)
        output_path = output_dir / f"{asset}_train_action_distribution.png"
        _plot_counts(asset=asset, counts=counts, output_path=output_path)
        saved_paths.append(output_path)
        LOGGER.info("Saved %s", output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Plot per-asset BUY/HOLD/SELL frequency distributions from labeled training data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/action_distribution"),
        help="Directory where output distribution plots are saved.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for plotting training action distributions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    saved_paths = plot_action_distributions(config_path=args.config, output_dir=args.output_dir)
    LOGGER.info("Completed. Saved %d plot file(s) to %s", len(saved_paths), args.output_dir)


if __name__ == "__main__":
    main()
