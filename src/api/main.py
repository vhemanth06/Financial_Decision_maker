from __future__ import annotations

import json
import logging
import os
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

from src.models.quant_xgboost import predict_actions_and_confidence
from src.models.rationale_rules import generate_rationale

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="CLEF-2026 Trading Agent API", version="1.0.0")

DEFAULT_FALLBACK_DECISION = "HOLD"
DEFAULT_FALLBACK_RATIONALE = "System fallback triggered due to data anomaly."
ACTION_TO_DECISION = {-1: "SELL", 0: "HOLD", 1: "BUY"}


class HistoryPricePoint(BaseModel):
    """Single historical price point used to reconstruct rolling indicators."""

    date: Optional[str] = None
    price: float


class PredictRequest(BaseModel):
    """Schema for daily prediction payloads.

    Supports both legacy flat payloads and CLEF nested per-asset payloads such as:
    - price: {"BTC": 67890.5}
    - news: {"BTC": ["..."]}
    - momentum: {"BTC": "bullish"}
    - 10k / 10q: {"BTC": "..."} or null
    - history_price: {"BTC": [{"date": "...", "price": ...}, ...]}
    """

    model_config = ConfigDict(
        extra="allow",
        validate_by_name=True,
    )

    # Legacy flat format.
    asset: Optional[str] = Field(default=None, description="Asset ticker, e.g., BTC")
    prices: Optional[float] = Field(default=None, description="Latest daily price value")
    momentum: Optional[str | int | float | dict[str, str | int | float | None]] = Field(
        default=None,
        description="Momentum signal (flat or nested by asset)",
    )
    news: Optional[str | list[str] | dict[str, str | list[str]]] = Field(
        default=None,
        description="News field (flat or nested by asset)",
    )
    filing_10k: Optional[str | list[str] | dict[str, str | list[str] | None]] = Field(
        default=None,
        description="Optional 10-K text (flat or nested by asset)",
    )
    filing_10q: Optional[str | list[str] | dict[str, str | list[str] | None]] = Field(
        default=None,
        description="Optional 10-Q text (flat or nested by asset)",
    )

    # Nested CLEF format.
    date: Optional[str] = Field(default=None, description="Payload date string")
    price: Optional[dict[str, float] | float] = Field(default=None, description="Nested or scalar price payload")
    symbol: Optional[dict[str, str | None] | str] = Field(default=None, description="Nested symbol payload")
    ten_k: Optional[dict[str, str | list[str] | None] | str | list[str]] = Field(
        default=None,
        alias="10k",
        description="Nested 10-K payload keyed by asset",
    )
    ten_q: Optional[dict[str, str | list[str] | None] | str | list[str]] = Field(
        default=None,
        alias="10q",
        description="Nested 10-Q payload keyed by asset",
    )
    history_price: Optional[dict[str, list[HistoryPricePoint]] | list[HistoryPricePoint]] = Field(
        default=None,
        description="Nested rolling price history keyed by asset",
    )


class PredictResponse(BaseModel):
    """Prediction response with strict action and short rationale."""

    decision: Literal["BUY", "HOLD", "SELL"]
    rationale: str


class InferenceService:
    """Runtime inference service that composes FinBERT, PCA, XGBoost, and rule-based rationale."""

    def __init__(self, config_path: Path) -> None:
        """Load all required artifacts for low-latency prediction.

        Args:
            config_path: Path to the configuration YAML.
        """
        self.config = self._load_config(config_path)
        self.paths_cfg = self.config["paths"]
        self.dataset_cfg = self.config["dataset"]
        self.features_cfg = self.config["features"]
        self.finbert_cfg = self.config["finbert"]
        self.xgb_cfg = self.config["xgboost"]
        self.rationale_cfg = self.config["rationale"]
        self._price_history_window = int(self.features_cfg.get("api_price_history_window", 60))
        self._price_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._price_history_window)
        )
        self._did_log_tabular_adjustment = False

        self._momentum_mapping = self.dataset_cfg["momentum_mapping"]
        self.runtime_policy = self._load_runtime_policy()

        self.xgb_model = self._load_xgb_model(Path(self.paths_cfg["xgb_model"]))
        self.pca_model = self._load_pca_model(Path(self.paths_cfg["pca_model"]))

        self.finbert_device = self._resolve_device(self.finbert_cfg["device"])
        self.finbert_tokenizer = AutoTokenizer.from_pretrained(self.finbert_cfg["model_name"])
        self.finbert_model = AutoModel.from_pretrained(
            self.finbert_cfg["model_name"],
            use_safetensors=True,
        ).to(self.finbert_device)
        if self.finbert_device.type == "cuda" and self.finbert_cfg["use_fp16_on_cuda"]:
            self.finbert_model = self.finbert_model.half()
        self.finbert_model.eval()

    def _load_runtime_policy(self) -> dict[str, float]:
        """Load optional calibrated decision policy from disk.

        Returns:
            Runtime policy dictionary used at inference.
        """
        policy = {
            "confidence_threshold": float(self.xgb_cfg["confidence_threshold"]),
            "directional_edge_threshold": float(self.xgb_cfg.get("directional_edge_threshold", 0.0)),
            "hold_probability_cap": float(self.xgb_cfg.get("hold_probability_cap", 1.0)),
        }

        policy_path = Path(self.paths_cfg.get("xgb_policy", "data/processed/xgb_policy.json"))
        if not policy_path.exists():
            return policy

        try:
            with policy_path.open("r", encoding="utf-8") as file_obj:
                loaded_policy = json.load(file_obj)

            if "confidence_threshold" in loaded_policy:
                policy["confidence_threshold"] = float(loaded_policy["confidence_threshold"])
            if "directional_edge_threshold" in loaded_policy:
                policy["directional_edge_threshold"] = float(loaded_policy["directional_edge_threshold"])
            if "hold_probability_cap" in loaded_policy:
                policy["hold_probability_cap"] = float(loaded_policy["hold_probability_cap"])

            LOGGER.info("Loaded runtime XGBoost policy from %s", policy_path)
        except Exception:
            LOGGER.exception("Failed to load policy file at %s; using config defaults.", policy_path)

        return policy

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        """Load YAML configuration file.

        Args:
            config_path: Config path.

        Returns:
            Parsed configuration dictionary.
        """
        with config_path.open("r", encoding="utf-8") as file_obj:
            return yaml.safe_load(file_obj)

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        """Resolve runtime device with auto fallback.

        Args:
            device_name: Requested device string.

        Returns:
            torch.device instance.
        """
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    @staticmethod
    def _load_xgb_model(model_path: Path) -> XGBClassifier:
        """Load trained XGBoost model from disk.

        Args:
            model_path: Serialized model path.

        Returns:
            Loaded XGBClassifier model.
        """
        model = XGBClassifier()
        model.load_model(str(model_path))
        return model

    @staticmethod
    def _load_pca_model(pca_path: Path) -> Any:
        """Load serialized PCA object from disk.

        Args:
            pca_path: Path to pickled PCA model.

        Returns:
            Loaded PCA object.
        """
        with pca_path.open("rb") as file_obj:
            return pickle.load(file_obj)

    @staticmethod
    def _coerce_optional_text(value: Any) -> str:
        """Normalize optional text payload fields.

        Args:
            value: Optional string or list input.

        Returns:
            Flattened text.
        """
        if value is None:
            return ""
        if isinstance(value, dict):
            parts = [str(item) for item in value.values() if item is not None]
            return " ".join(parts)
        if isinstance(value, list):
            return " ".join(str(item) for item in value if item is not None)
        return str(value)

    @staticmethod
    def _first_mapping_key(value: Any) -> Optional[str]:
        """Return first key from a mapping-like payload when available."""
        if isinstance(value, dict) and value:
            return str(next(iter(value.keys())))
        return None

    @staticmethod
    def _mapping_value_for_asset(value: Any, asset_key: str) -> Any:
        """Extract per-asset value from a dict payload with safe fallbacks."""
        if not isinstance(value, dict):
            return value

        if asset_key in value:
            return value[asset_key]

        upper_index = {str(key).upper(): key for key in value.keys()}
        if asset_key.upper() in upper_index:
            return value[upper_index[asset_key.upper()]]

        return next(iter(value.values())) if value else None

    def _resolve_asset(self, payload: PredictRequest) -> str:
        """Resolve asset key from flat or nested input payload."""
        if payload.asset is not None and str(payload.asset).strip():
            return str(payload.asset).strip().upper()

        for candidate in [
            payload.symbol,
            payload.price,
            payload.news,
            payload.momentum,
            payload.history_price,
            payload.ten_k,
            payload.ten_q,
            payload.filing_10k,
            payload.filing_10q,
        ]:
            first_key = self._first_mapping_key(candidate)
            if first_key is not None and first_key.strip():
                return first_key.strip().upper()

        raise ValueError("Unable to infer asset from payload. Provide asset or nested keyed input.")

    def _resolve_price(self, payload: PredictRequest, asset_key: str) -> float:
        """Resolve current price from flat or nested input payload."""
        if payload.prices is not None:
            return float(payload.prices)

        if payload.price is None:
            raise ValueError("Missing price/prices field in payload")

        value = self._mapping_value_for_asset(payload.price, asset_key)
        if value is None:
            raise ValueError("Price payload does not contain a usable value")
        return float(value)

    def _resolve_news_text(self, payload: PredictRequest, asset_key: str) -> str:
        """Resolve daily news text from flat or nested payload."""
        news_value = self._mapping_value_for_asset(payload.news, asset_key)
        return self._coerce_optional_text(news_value)

    def _resolve_filing_text(self, value: Any, asset_key: str) -> str:
        """Resolve filing text from flat or nested payload."""
        return self._coerce_optional_text(self._mapping_value_for_asset(value, asset_key))

    def _resolve_momentum(self, payload: PredictRequest, asset_key: str) -> Optional[str | int | float]:
        """Resolve momentum value from flat or nested payload."""
        momentum_value = self._mapping_value_for_asset(payload.momentum, asset_key)
        if momentum_value is None:
            return None
        if isinstance(momentum_value, (str, int, float)):
            return momentum_value
        return str(momentum_value)

    def _resolve_history_prices(self, payload: PredictRequest, asset_key: str) -> Optional[list[float]]:
        """Resolve optional historical prices from nested payload."""
        if payload.history_price is None:
            return None

        history_value = self._mapping_value_for_asset(payload.history_price, asset_key)
        if history_value is None:
            return None

        parsed_prices: list[float] = []
        if isinstance(history_value, list):
            for item in history_value:
                if isinstance(item, HistoryPricePoint):
                    parsed_prices.append(float(item.price))
                elif isinstance(item, dict) and "price" in item:
                    parsed_prices.append(float(item["price"]))
                elif isinstance(item, (int, float)):
                    parsed_prices.append(float(item))

        if not parsed_prices:
            return None
        return parsed_prices

    def _to_momentum_numeric(self, momentum: Optional[str | int | float]) -> int:
        """Convert momentum payload value into configured numeric representation.

        Args:
            momentum: Raw momentum value.

        Returns:
            Encoded momentum integer.
        """
        if momentum is None:
            return int(self._momentum_mapping.get("missing", 0))

        if isinstance(momentum, int):
            return momentum

        if isinstance(momentum, float):
            return int(momentum)

        mapped = self._momentum_mapping.get(str(momentum).strip().lower())
        if mapped is None:
            return int(self._momentum_mapping.get("missing", 0))
        return int(mapped)

    def _build_text(self, payload: PredictRequest, asset_key: str) -> str:
        """Construct model input text from news and filing fields.

        Args:
            payload: Incoming predict payload.

        Returns:
            Single normalized text string.
        """
        filing_10k_value = payload.ten_k if payload.ten_k is not None else payload.filing_10k
        filing_10q_value = payload.ten_q if payload.ten_q is not None else payload.filing_10q

        text = " ".join(
            [
                self._resolve_news_text(payload, asset_key),
                self._resolve_filing_text(filing_10k_value, asset_key),
                self._resolve_filing_text(filing_10q_value, asset_key),
            ]
        )
        text = " ".join(text.split())
        return text

    @staticmethod
    def _compute_rsi(prices: list[float], window: int = 14) -> float:
        """Compute RSI from in-memory rolling prices with safe fallbacks."""
        if len(prices) < 2:
            return 50.0

        price_array = np.asarray(prices, dtype=np.float64)
        deltas = np.diff(price_array)
        if deltas.size == 0:
            return 50.0

        gains = np.maximum(deltas, 0.0)
        losses = np.maximum(-deltas, 0.0)

        selected_window = min(window, deltas.size)
        avg_gain = float(np.mean(gains[-selected_window:]))
        avg_loss = float(np.mean(losses[-selected_window:]))

        if avg_gain == 0.0 and avg_loss == 0.0:
            return 50.0
        if avg_loss == 0.0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.clip(rsi, 0.0, 100.0))

    @staticmethod
    def _compute_ma_ratio(prices: list[float], short_window: int = 10, long_window: int = 30) -> float:
        """Compute short/long moving average ratio with neutral fallback."""
        if len(prices) < 2:
            return 1.0

        price_array = np.asarray(prices, dtype=np.float64)
        short_len = min(short_window, len(price_array))
        long_len = min(long_window, len(price_array))

        short_ma = float(np.mean(price_array[-short_len:]))
        long_ma = float(np.mean(price_array[-long_len:]))
        if long_ma == 0.0:
            return 1.0
        return float(short_ma / long_ma)

    @staticmethod
    def _compute_rolling_volatility(prices: list[float], window: int = 20) -> float:
        """Compute rolling return volatility with robust low-sample fallback."""
        if len(prices) < 3:
            return 0.0

        price_array = np.asarray(prices, dtype=np.float64)
        returns = np.diff(price_array) / np.where(price_array[:-1] == 0.0, np.nan, price_array[:-1])
        returns = returns[~np.isnan(returns)]

        if returns.size < 2:
            return 0.0

        selected_window = min(window, returns.size)
        sample = returns[-selected_window:]
        if sample.size < 2:
            return 0.0
        return float(np.std(sample, ddof=1))

    def _build_tabular_features(
        self,
        asset_key: str,
        price_value: float,
        momentum_numeric: int,
        history_seed_prices: Optional[list[float]],
    ) -> np.ndarray:
        """Build tabular features in the same order used by model training."""
        history = self._price_history[asset_key]
        if history_seed_prices:
            history.clear()
            history.extend(float(price) for price in history_seed_prices)
        history.append(price_value)
        history_prices = list(history)

        feature_values = {
            "prices": price_value,
            "momentum_numeric": float(momentum_numeric),
            "rsi_14d": self._compute_rsi(history_prices, window=14),
            "ma_ratio_10_30": self._compute_ma_ratio(history_prices, short_window=10, long_window=30),
            "vol_20d": self._compute_rolling_volatility(history_prices, window=20),
        }

        tabular_columns = [str(column_name) for column_name in self.features_cfg["tabular_columns"]]
        tabular_values = [float(feature_values.get(column_name, 0.0)) for column_name in tabular_columns]
        return np.array([tabular_values], dtype=np.float32)

    def _encode_single_text(self, text: str) -> np.ndarray:
        """Encode one text record using FinBERT and project to PCA space.

        Args:
            text: Input text.

        Returns:
            PCA-reduced embedding row with shape (1, n_components).
        """
        encoded = self.finbert_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.finbert_cfg["max_length"],
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.finbert_device) for name, tensor in encoded.items()}

        with torch.no_grad():
            outputs = self.finbert_model(**encoded)
            if getattr(outputs, "pooler_output", None) is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]

        dense_embedding = pooled.detach().float().cpu().numpy().astype(np.float32)
        reduced_embedding = self.pca_model.transform(dense_embedding).astype(np.float32)
        return reduced_embedding

    def predict(self, payload: PredictRequest) -> tuple[str, str]:
        """Run full late-fusion inference and rationale generation.

        Args:
            payload: Daily payload.

        Returns:
            Tuple of decision string and rationale text.
        """
        asset_key = self._resolve_asset(payload)
        price_value = self._resolve_price(payload, asset_key)
        momentum_raw = self._resolve_momentum(payload, asset_key)
        history_seed_prices = self._resolve_history_prices(payload, asset_key)

        combined_text = self._build_text(payload, asset_key)
        momentum_numeric = self._to_momentum_numeric(momentum_raw)

        reduced_embedding = self._encode_single_text(combined_text)
        tabular = self._build_tabular_features(
            asset_key=asset_key,
            price_value=price_value,
            momentum_numeric=momentum_numeric,
            history_seed_prices=history_seed_prices,
        )

        expected_total_features = int(self.xgb_model.n_features_in_)
        embedding_width = int(reduced_embedding.shape[1])
        required_tabular_width = expected_total_features - embedding_width
        if required_tabular_width <= 0:
            raise ValueError(
                "Invalid model/input configuration: expected total feature count "
                f"{expected_total_features} is smaller than embedding width {embedding_width}."
            )

        if tabular.shape[1] != required_tabular_width:
            if not self._did_log_tabular_adjustment:
                LOGGER.warning(
                    "Adjusting tabular feature width from %d to %d to match model expectations.",
                    tabular.shape[1],
                    required_tabular_width,
                )
                self._did_log_tabular_adjustment = True

            if tabular.shape[1] > required_tabular_width:
                tabular = tabular[:, :required_tabular_width]
            else:
                pad_width = required_tabular_width - tabular.shape[1]
                padding = np.zeros((tabular.shape[0], pad_width), dtype=np.float32)
                tabular = np.hstack([tabular, padding])

        features = np.hstack([tabular, reduced_embedding]).astype(np.float32)

        actions, confidences = predict_actions_and_confidence(
            model=self.xgb_model,
            features=features,
            confidence_threshold=float(self.runtime_policy["confidence_threshold"]),
            directional_edge_threshold=float(self.runtime_policy["directional_edge_threshold"]),
            hold_probability_cap=float(self.runtime_policy["hold_probability_cap"]),
        )
        action_value = int(actions[0])
        decision = ACTION_TO_DECISION.get(action_value, DEFAULT_FALLBACK_DECISION)
        confidence = float(confidences[0])

        rationale = generate_rationale(
            decision=decision,
            asset=asset_key,
            confidence=confidence,
            momentum_numeric=momentum_numeric,
            day_text=combined_text,
            max_words=int(self.rationale_cfg["max_words"]),
            context_char_limit=int(self.rationale_cfg["context_char_limit"]),
            confidence_low=float(self.rationale_cfg["confidence_low"]),
            confidence_high=float(self.rationale_cfg["confidence_high"]),
            positive_keywords=[str(item) for item in self.rationale_cfg["positive_keywords"]],
            negative_keywords=[str(item) for item in self.rationale_cfg["negative_keywords"]],
        )

        return decision, rationale


def _resolve_config_path() -> Path:
    """Resolve API config path from environment or project-relative fallback."""
    env_path = os.getenv("CLEF_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_fallback_response() -> dict[str, str]:
    """Resolve fallback decision/rationale from config when available.

    Returns:
        Fallback response dictionary.
    """
    fallback = {
        "decision": DEFAULT_FALLBACK_DECISION,
        "rationale": DEFAULT_FALLBACK_RATIONALE,
    }

    try:
        config_path = _resolve_config_path()
        with config_path.open("r", encoding="utf-8") as file_obj:
            config = yaml.safe_load(file_obj)
        fallback["decision"] = str(config.get("api", {}).get("fallback_decision", fallback["decision"]))
        fallback["rationale"] = str(config.get("api", {}).get("fallback_rationale", fallback["rationale"]))
    except Exception:
        pass

    return fallback


_SERVICE: Optional[InferenceService] = None


def _get_service() -> InferenceService:
    """Build and cache the singleton inference service.

    Returns:
        Initialized InferenceService object.
    """
    global _SERVICE
    if _SERVICE is None:
        config_path = _resolve_config_path()
        _SERVICE = InferenceService(config_path=config_path)
    return _SERVICE


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict one decision and one rationale with full circuit-breaker fallback.

    Args:
        payload: Daily JSON payload for one asset.

    Returns:
        Decision and rationale response.
    """
    try:
        service = _get_service()
        decision, rationale = service.predict(payload)
        return PredictResponse(decision=decision, rationale=rationale)
    except Exception:
        LOGGER.exception("Prediction failure; returning fallback HOLD response.")
        fallback = _load_fallback_response()
        return PredictResponse(
            decision=fallback["decision"],
            rationale=fallback["rationale"],
        )


@app.post("/", response_model=PredictResponse, include_in_schema=False)
def predict_root(payload: PredictRequest) -> PredictResponse:
    """Backward-compatible root POST alias for clients posting to /."""
    return predict(payload)
