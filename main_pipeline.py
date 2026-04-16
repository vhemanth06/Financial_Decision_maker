from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.clean_data import build_consolidated_dataframe
from src.data.fetch_assets import fetch_assets
from src.evaluation.cross_val import build_cpcv_from_config
from src.evaluation.metrics import evaluate_trading_performance
from src.features.labels import build_labeled_dataset
from src.features.text_encoder import build_finbert_pca_embeddings
from src.models.quant_xgboost import (
    build_feature_matrix,
    predict_actions_with_threshold,
    train_and_save_xgboost,
    train_xgb_classifier,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logger for the end-to-end pipeline.

    Centralized logging is required so every phase emits traceable diagnostics during
    long-running training jobs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Path to config file.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def set_global_seed(config: dict[str, Any]) -> None:
    """Set reproducibility seeds across Python, NumPy, and PyTorch.

    Args:
        config: Full project configuration dictionary.
    """
    random.seed(int(config["seed"]["numpy"]))
    np.random.seed(int(config["seed"]["numpy"]))

    torch.manual_seed(int(config["seed"]["torch"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]["torch"]))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_cpcv_evaluation(config: dict[str, Any]) -> dict[str, float]:
    """Run CPCV Sharpe evaluation using fee-aware returns.

    Args:
        config: Full project configuration dictionary.

    Returns:
        Summary dictionary with mean/std Sharpe and split count.
    """
    paths_cfg = config["paths"]
    features_cfg = config["features"]
    xgb_cfg = config["xgboost"]
    evaluation_cfg = config["evaluation"]

    labeled_frame = pd.read_parquet(Path(paths_cfg["labeled_data"]))
    reduced_embeddings = np.load(Path(paths_cfg["finbert_pca_embeddings"]))

    all_features, all_targets_encoded, valid_mask = build_feature_matrix(
        frame=labeled_frame,
        reduced_embeddings=reduced_embeddings,
        tabular_columns=features_cfg["tabular_columns"],
        target_column=features_cfg["target_column"],
    )

    valid_frame = labeled_frame.loc[valid_mask].reset_index(drop=True)
    forward_returns = pd.to_numeric(valid_frame["future_return"], errors="coerce").fillna(0.0).to_numpy()

    splitter = build_cpcv_from_config(config)
    sharpe_scores: list[float] = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(all_features), start=1):
        split_seed = int(config["seed"]["xgboost"]) + split_id
        model = train_xgb_classifier(
            features=all_features[train_idx],
            targets_encoded=all_targets_encoded[train_idx],
            xgb_cfg=xgb_cfg,
            seed=split_seed,
        )

        actions = predict_actions_with_threshold(
            model=model,
            features=all_features[test_idx],
            confidence_threshold=float(xgb_cfg["confidence_threshold"]),
        )

        metrics = evaluate_trading_performance(
            actions=actions,
            market_returns=forward_returns[test_idx],
            transaction_fee_bps=float(evaluation_cfg["transaction_fee_bps"]),
            annualization_factor=int(evaluation_cfg["annualization_factor"]),
        )

        sharpe = float(metrics["sharpe_ratio"])
        sharpe_scores.append(sharpe)
        LOGGER.info("CPCV split %d Sharpe: %.6f", split_id, sharpe)

    if not sharpe_scores:
        return {
            "mean_sharpe": 0.0,
            "std_sharpe": 0.0,
            "num_splits": 0.0,
        }

    return {
        "mean_sharpe": float(np.mean(sharpe_scores)),
        "std_sharpe": float(np.std(sharpe_scores, ddof=1)) if len(sharpe_scores) > 1 else 0.0,
        "num_splits": float(len(sharpe_scores)),
    }


def run_pipeline(config_path: Path) -> None:
    """Execute end-to-end training and artifact generation in strict phase order.

    Args:
        config_path: Path to configs/config.yaml.
    """
    config = load_config(config_path)
    set_global_seed(config)

    LOGGER.info("Phase 2.1 | Fetching raw assets from Hugging Face datasets")
    fetch_assets(config)

    LOGGER.info("Phase 2.2 | Cleaning raw files and building consolidated dataset")
    build_consolidated_dataframe(config)

    LOGGER.info("Phase 3.1 | Generating dynamic-threshold labels")
    build_labeled_dataset(config)

    LOGGER.info("Phase 3.2 | Generating FinBERT embeddings and PCA projections")
    build_finbert_pca_embeddings(config)

    LOGGER.info("Phase 5 | Running CPCV Sharpe evaluation")
    cpcv_summary = run_cpcv_evaluation(config)
    LOGGER.info(
        "CPCV summary | mean_sharpe=%.6f std_sharpe=%.6f splits=%d",
        cpcv_summary["mean_sharpe"],
        cpcv_summary["std_sharpe"],
        int(cpcv_summary["num_splits"]),
    )

    LOGGER.info("Phase 4 | Training final XGBoost model on full training set")
    train_and_save_xgboost(config)

    LOGGER.info("Pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Run CLEF-2026 trading-agent training pipeline.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """Entry point for end-to-end training pipeline."""
    configure_logging()
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
