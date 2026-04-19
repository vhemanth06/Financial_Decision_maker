from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {class_id: label for label, class_id in LABEL_TO_CLASS.items()}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _encode_targets(raw_targets: np.ndarray) -> np.ndarray:
    """Encode targets from {-1,0,1} to XGBoost class IDs {0,1,2}.

    Args:
        raw_targets: Raw target array.

    Returns:
        Encoded class IDs.

    Raises:
        ValueError: If unseen class labels are detected.
    """
    unique_labels = set(np.unique(raw_targets).tolist())
    unsupported = unique_labels.difference(LABEL_TO_CLASS.keys())
    if unsupported:
        raise ValueError(f"Unsupported labels detected: {unsupported}")

    return np.array([LABEL_TO_CLASS[int(label)] for label in raw_targets], dtype=np.int64)


def _decode_classes(class_ids: np.ndarray) -> np.ndarray:
    """Decode XGBoost class IDs {0,1,2} back to {-1,0,1} actions.

    Args:
        class_ids: Predicted class ID array.

    Returns:
        Decoded action labels.
    """
    return np.array([CLASS_TO_LABEL[int(class_id)] for class_id in class_ids], dtype=np.int8)


def _compute_balanced_sample_weights(targets_encoded: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights for multiclass training.

    Args:
        targets_encoded: Encoded target array with class IDs.

    Returns:
        Per-sample weight vector.
    """
    if len(targets_encoded) == 0:
        return np.array([], dtype=np.float32)

    class_ids, class_counts = np.unique(targets_encoded, return_counts=True)
    if len(class_ids) <= 1:
        return np.ones(len(targets_encoded), dtype=np.float32)

    total = float(np.sum(class_counts))
    class_count = float(len(class_ids))
    class_to_weight = {
        int(class_id): total / (class_count * float(count))
        for class_id, count in zip(class_ids, class_counts)
        if count > 0
    }
    return np.array([class_to_weight[int(class_id)] for class_id in targets_encoded], dtype=np.float32)


def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_ma_ratio(series: pd.Series, short_window: int = 10, long_window: int = 30) -> pd.Series:
    """Calculate moving average ratio."""
    short_ma = series.rolling(window=short_window).mean()
    long_ma = series.rolling(window=long_window).mean()
    return short_ma / long_ma


def _calculate_rolling_vol(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility of returns."""
    return series.pct_change().rolling(window=window).std()


def actions_from_probabilities(
    probabilities: np.ndarray,
    confidence_threshold: float,
    directional_edge_threshold: float = 0.0,
    hold_probability_cap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert class probabilities into {-1,0,1} actions with policy guards.

    Args:
        probabilities: Predicted class probabilities with shape (n_samples, 3).
        confidence_threshold: Minimum max class probability to allow directional action.
        directional_edge_threshold: Minimum absolute edge between BUY and SELL probabilities.
        hold_probability_cap: If HOLD probability exceeds this cap, action is forced to HOLD.

    Returns:
        Tuple of (actions, best_probabilities).
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if probabilities.shape[1] != len(LABEL_TO_CLASS):
        raise ValueError(
            "probabilities must have one column per class "
            f"({len(LABEL_TO_CLASS)}), got {probabilities.shape[1]}"
        )

    best_class_ids = np.argmax(probabilities, axis=1)
    best_probabilities = np.max(probabilities, axis=1).astype(np.float32)
    actions = _decode_classes(best_class_ids)

    low_confidence_mask = best_probabilities < float(confidence_threshold)

    if hold_probability_cap < 1.0:
        hold_class_id = LABEL_TO_CLASS[0]
        low_confidence_mask = low_confidence_mask | (probabilities[:, hold_class_id] >= float(hold_probability_cap))

    if directional_edge_threshold > 0.0:
        buy_class_id = LABEL_TO_CLASS[1]
        sell_class_id = LABEL_TO_CLASS[-1]
        directional_edge = np.abs(probabilities[:, buy_class_id] - probabilities[:, sell_class_id])
        low_confidence_mask = low_confidence_mask | (directional_edge < float(directional_edge_threshold))

    actions[low_confidence_mask] = 0
    return actions, best_probabilities


def build_feature_matrix(
    frame: pd.DataFrame,
    reduced_embeddings: np.ndarray,
    tabular_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct late-fusion matrix from tabular and PCA text embeddings.

    Args:
        frame: Labeled dataframe.
        reduced_embeddings: PCA-compressed FinBERT matrix.
        tabular_columns: Ordered list of tabular feature columns.
        target_column: Target column name.

    Returns:
        A tuple of (train_features, train_targets_encoded, valid_mask).

    Raises:
        ValueError: If row counts between dataframe and embeddings do not match.
    """
    if len(frame) != reduced_embeddings.shape[0]:
        raise ValueError(
            "Row mismatch between dataframe and embeddings: "
            f"{len(frame)} != {reduced_embeddings.shape[0]}. "
            "Regenerate text embeddings to ensure artifacts are aligned with the current labeled dataset."
        )

    # --- Start: Technical Indicator Generation ---
    grouped_prices = frame.groupby("asset")["prices"]
    frame["rsi_14d"] = grouped_prices.transform(_calculate_rsi)
    frame["ma_ratio_10_30"] = grouped_prices.transform(_calculate_ma_ratio)
    frame["vol_20d"] = grouped_prices.transform(_calculate_rolling_vol)
    # --- End: Technical Indicator Generation ---

    tabular = (
        frame[tabular_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    reduced_embeddings = reduced_embeddings.astype(np.float32)
    all_features = np.hstack([tabular, reduced_embeddings])

    target_series = pd.to_numeric(frame[target_column], errors="coerce")
    valid_mask = target_series.isin([-1, 0, 1]) & target_series.notna()

    train_features = all_features[valid_mask.to_numpy()]
    train_targets_raw = target_series.loc[valid_mask].astype(int).to_numpy(dtype=np.int64)
    train_targets_encoded = _encode_targets(train_targets_raw)

    return train_features, train_targets_encoded, valid_mask.to_numpy()


def train_xgb_classifier(
    features: np.ndarray,
    targets_encoded: np.ndarray,
    xgb_cfg: dict[str, Any],
    seed: int,
) -> XGBClassifier:
    """Train a multiclass XGBoost classifier for action prediction.

    Args:
        features: Late-fusion feature matrix.
        targets_encoded: Encoded class IDs.
        xgb_cfg: XGBoost configuration dictionary.
        seed: Random seed for deterministic training.

    Returns:
        Trained XGBClassifier instance.
    """
    model = XGBClassifier(
        objective=xgb_cfg["objective"],
        num_class=xgb_cfg["num_class"],
        eval_metric=xgb_cfg["eval_metric"],
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        reg_lambda=xgb_cfg["reg_lambda"],
        min_child_weight=xgb_cfg["min_child_weight"],
        gamma=xgb_cfg["gamma"],
        tree_method=xgb_cfg["tree_method"],
        n_jobs=xgb_cfg["n_jobs"],
        random_state=seed,
        verbosity=0,
    )

    sample_weight = None
    if bool(xgb_cfg.get("use_class_balancing", True)):
        sample_weight = _compute_balanced_sample_weights(targets_encoded)

    model.fit(features, targets_encoded, sample_weight=sample_weight)
    return model


def predict_actions_and_confidence(
    model: XGBClassifier,
    features: np.ndarray,
    confidence_threshold: float,
    directional_edge_threshold: float = 0.0,
    hold_probability_cap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict actions and expose confidence after HOLD thresholding.

    Args:
        model: Trained XGBoost classifier.
        features: Feature matrix for inference.
        confidence_threshold: Min class probability required to trust a trade.
        directional_edge_threshold: Min BUY-vs-SELL probability edge for directional actions.
        hold_probability_cap: Force HOLD when HOLD probability is above this cap.

    Returns:
        Tuple of (actions, best_probabilities) where actions are in {-1, 0, 1}.
    """
    probabilities = model.predict_proba(features)
    return actions_from_probabilities(
        probabilities=probabilities,
        confidence_threshold=confidence_threshold,
        directional_edge_threshold=directional_edge_threshold,
        hold_probability_cap=hold_probability_cap,
    )


def predict_actions_with_threshold(
    model: XGBClassifier,
    features: np.ndarray,
    confidence_threshold: float,
    directional_edge_threshold: float = 0.0,
    hold_probability_cap: float = 1.0,
) -> np.ndarray:
    """Predict discrete actions with confidence-threshold HOLD override.

    Args:
        model: Trained XGBoost classifier.
        features: Feature matrix for inference.
        confidence_threshold: Min class probability required to trust a trade.
        directional_edge_threshold: Min BUY-vs-SELL probability edge for directional actions.
        hold_probability_cap: Force HOLD when HOLD probability is above this cap.

    Returns:
        Array of decoded actions in {-1, 0, 1}.
    """
    actions, _ = predict_actions_and_confidence(
        model=model,
        features=features,
        confidence_threshold=confidence_threshold,
        directional_edge_threshold=directional_edge_threshold,
        hold_probability_cap=hold_probability_cap,
    )
    return actions


def train_and_save_xgboost(config: dict[str, Any]) -> XGBClassifier:
    """Train XGBoost from configured artifacts and save model to disk.

    Args:
        config: Full project configuration.

    Returns:
        Trained XGBClassifier.
    """
    paths_cfg = config["paths"]
    features_cfg = config["features"]
    xgb_cfg = config["xgboost"]

    frame = pd.read_parquet(Path(paths_cfg["labeled_data"]))
    reduced_embeddings = np.load(Path(paths_cfg["finbert_pca_embeddings"]))

    train_features, train_targets_encoded, _ = build_feature_matrix(
        frame=frame,
        reduced_embeddings=reduced_embeddings,
        tabular_columns=features_cfg["tabular_columns"],
        target_column=features_cfg["target_column"],
    )

    model = train_xgb_classifier(
        features=train_features,
        targets_encoded=train_targets_encoded,
        xgb_cfg=xgb_cfg,
        seed=config["seed"]["xgboost"],
    )

    model_path = Path(paths_cfg["xgb_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    train_actions = predict_actions_with_threshold(
        model=model,
        features=train_features,
        confidence_threshold=xgb_cfg["confidence_threshold"],
        directional_edge_threshold=float(xgb_cfg.get("directional_edge_threshold", 0.0)),
        hold_probability_cap=float(xgb_cfg.get("hold_probability_cap", 1.0)),
    )

    hold_share = float(np.mean(train_actions == 0)) if len(train_actions) else 0.0
    LOGGER.info(
        "Saved XGBoost model -> %s | train_samples=%d | hold_share=%.4f",
        model_path,
        len(train_targets_encoded),
        hold_share,
    )

    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Train late-fusion XGBoost model.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """Run XGBoost model training as a standalone script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    train_and_save_xgboost(config)


if __name__ == "__main__":
    main()
