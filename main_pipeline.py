from __future__ import annotations

import argparse
import json
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
    actions_from_probabilities,
    build_feature_matrix,
    train_and_save_xgboost,
    train_xgb_classifier,
)

LOGGER = logging.getLogger(__name__)


# Configure root logger for the pipeline.
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Load project configuration from YAML.
def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


# Set reproducibility seeds.
def set_global_seed(config: dict[str, Any]) -> None:
    random.seed(int(config["seed"]["numpy"]))
    np.random.seed(int(config["seed"]["numpy"]))

    torch.manual_seed(int(config["seed"]["torch"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]["torch"]))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Run CPCV Sharpe evaluation.
def run_cpcv_evaluation(config: dict[str, Any]) -> dict[str, float]:
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

    threshold_grid = [
        float(value)
        for value in xgb_cfg.get("confidence_threshold_grid", [])
    ]
    threshold_grid.append(float(xgb_cfg["confidence_threshold"]))
    threshold_grid = sorted({threshold for threshold in threshold_grid if 0.0 <= threshold <= 1.0})
    if not threshold_grid:
        threshold_grid = [float(xgb_cfg["confidence_threshold"])]

    directional_edge_threshold = float(xgb_cfg.get("directional_edge_threshold", 0.0))
    hold_probability_cap = float(xgb_cfg.get("hold_probability_cap", 1.0))

    splitter = build_cpcv_from_config(config)
    sharpe_scores_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in threshold_grid}
    hold_share_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in threshold_grid}

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(all_features), start=1):
        split_seed = int(config["seed"]["xgboost"]) + split_id
        model = train_xgb_classifier(
            features=all_features[train_idx],
            targets_encoded=all_targets_encoded[train_idx],
            xgb_cfg=xgb_cfg,
            seed=split_seed,
        )

        split_probabilities = model.predict_proba(all_features[test_idx])
        for threshold in threshold_grid:
            actions, _ = actions_from_probabilities(
                probabilities=split_probabilities,
                confidence_threshold=float(threshold),
                directional_edge_threshold=directional_edge_threshold,
                hold_probability_cap=hold_probability_cap,
            )

            metrics = evaluate_trading_performance(
                actions=actions,
                market_returns=forward_returns[test_idx],
                transaction_fee_bps=float(evaluation_cfg["transaction_fee_bps"]),
                annualization_factor=int(evaluation_cfg["annualization_factor"]),
            )

            sharpe_scores_by_threshold[threshold].append(float(metrics["sharpe_ratio"]))
            hold_share_by_threshold[threshold].append(float(np.mean(actions == 0)) if len(actions) else 0.0)

    non_empty_thresholds = {
        threshold: values
        for threshold, values in sharpe_scores_by_threshold.items()
        if values
    }
    if not non_empty_thresholds:
        return {
            "mean_sharpe": 0.0,
            "std_sharpe": 0.0,
            "num_splits": 0.0,
            "best_confidence_threshold": float(xgb_cfg["confidence_threshold"]),
            "best_mean_sharpe": 0.0,
            "mean_hold_share": 0.0,
        }

    mean_sharpe_by_threshold = {
        threshold: float(np.mean(values))
        for threshold, values in non_empty_thresholds.items()
    }
    best_threshold = max(
        mean_sharpe_by_threshold,
        key=lambda threshold: (mean_sharpe_by_threshold[threshold], -threshold),
    )

    for threshold in threshold_grid:
        threshold_sharpes = sharpe_scores_by_threshold.get(threshold, [])
        threshold_holds = hold_share_by_threshold.get(threshold, [])
        if not threshold_sharpes:
            continue
        LOGGER.info(
            "CPCV threshold %.3f | mean_sharpe=%.6f mean_hold_share=%.4f splits=%d",
            threshold,
            float(np.mean(threshold_sharpes)),
            float(np.mean(threshold_holds)) if threshold_holds else 0.0,
            len(threshold_sharpes),
        )

    selected_sharpes = sharpe_scores_by_threshold[best_threshold]
    selected_holds = hold_share_by_threshold[best_threshold]
    return {
        "mean_sharpe": float(np.mean(selected_sharpes)),
        "std_sharpe": float(np.std(selected_sharpes, ddof=1)) if len(selected_sharpes) > 1 else 0.0,
        "num_splits": float(len(selected_sharpes)),
        "best_confidence_threshold": float(best_threshold),
        "best_mean_sharpe": float(mean_sharpe_by_threshold[best_threshold]),
        "mean_hold_share": float(np.mean(selected_holds)) if selected_holds else 0.0,
    }


# Persist calibrated inference policy values.
def save_xgb_policy(config: dict[str, Any], cpcv_summary: dict[str, float]) -> None:
    paths_cfg = config["paths"]
    xgb_cfg = config["xgboost"]

    policy_path = Path(paths_cfg.get("xgb_policy", "data/processed/xgb_policy.json"))
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    policy_payload = {
        "confidence_threshold": float(
            cpcv_summary.get("best_confidence_threshold", xgb_cfg["confidence_threshold"])
        ),
        "directional_edge_threshold": float(xgb_cfg.get("directional_edge_threshold", 0.0)),
        "hold_probability_cap": float(xgb_cfg.get("hold_probability_cap", 1.0)),
        "sharpe_ratio": float(cpcv_summary.get("mean_sharpe", 0.0)),
        "sharpe_std": float(cpcv_summary.get("std_sharpe", 0.0)),
        "mean_hold_share": float(cpcv_summary.get("mean_hold_share", 0.0)),
        "objective": float(cpcv_summary.get("best_mean_sharpe", 0.0)),
    }

    with policy_path.open("w", encoding="utf-8") as file_obj:
        json.dump(policy_payload, file_obj, indent=2)
    LOGGER.info("Saved XGBoost policy -> %s", policy_path)


# Execute end-to-end training and artifact generation in strict phase order.
def run_pipeline(config_path: Path) -> None:
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
        "CPCV summary | mean_sharpe=%.6f std_sharpe=%.6f splits=%d tuned_threshold=%.3f",
        cpcv_summary["mean_sharpe"],
        cpcv_summary["std_sharpe"],
        int(cpcv_summary["num_splits"]),
        cpcv_summary["best_confidence_threshold"],
    )

    config["xgboost"]["confidence_threshold"] = float(cpcv_summary["best_confidence_threshold"])
    LOGGER.info(
        "Using tuned confidence threshold %.3f for final training and policy export",
        config["xgboost"]["confidence_threshold"],
    )
    save_xgb_policy(config, cpcv_summary)

    LOGGER.info("Phase 4 | Training final XGBoost model on full training set")
    train_and_save_xgboost(config)

    LOGGER.info("Pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLEF-2026 trading-agent training pipeline.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
