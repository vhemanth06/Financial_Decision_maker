from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

from src.models.quant_xgboost import predict_actions_and_confidence
from src.models.rationale_rules import generate_rationale

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="CLEF-2026 Trading Agent API", version="1.0.0")

DEFAULT_FALLBACK_DECISION = "HOLD"
DEFAULT_FALLBACK_RATIONALE = "System fallback triggered due to data anomaly."
ACTION_TO_DECISION = {-1: "SELL", 0: "HOLD", 1: "BUY"}


class PredictRequest(BaseModel):
    """Schema for daily prediction payloads.

    Fields mirror the expected CLEF daily payload while allowing extra keys to keep
    endpoint parsing resilient to benign schema extensions.
    """

    class Config:
        """Allow payload schema extensions without rejecting requests."""

        extra = "allow"

    asset: str = Field(..., description="Asset ticker, e.g., BTC")
    prices: float = Field(..., description="Latest daily price value")
    momentum: Optional[str | int | float] = Field(default=None, description="Momentum category or numeric state")
    news: str = Field(default="", description="Daily news summary")
    filing_10k: Optional[str | list[str]] = Field(default=None, description="Optional 10-K text")
    filing_10q: Optional[str | list[str]] = Field(default=None, description="Optional 10-Q text")


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

        self._momentum_mapping = self.dataset_cfg["momentum_mapping"]

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
    def _coerce_optional_text(value: Optional[str | list[str]]) -> str:
        """Normalize optional text payload fields.

        Args:
            value: Optional string or list input.

        Returns:
            Flattened text.
        """
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(str(item) for item in value if item is not None)
        return str(value)

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

    def _build_text(self, payload: PredictRequest) -> str:
        """Construct model input text from news and filing fields.

        Args:
            payload: Incoming predict payload.

        Returns:
            Single normalized text string.
        """
        text = " ".join(
            [
                payload.news,
                self._coerce_optional_text(payload.filing_10k),
                self._coerce_optional_text(payload.filing_10q),
            ]
        )
        text = " ".join(text.split())
        return text

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
        combined_text = self._build_text(payload)
        momentum_numeric = self._to_momentum_numeric(payload.momentum)

        reduced_embedding = self._encode_single_text(combined_text)
        tabular = np.array([[float(payload.prices), float(momentum_numeric)]], dtype=np.float32)
        features = np.hstack([tabular, reduced_embedding]).astype(np.float32)

        actions, confidences = predict_actions_and_confidence(
            model=self.xgb_model,
            features=features,
            confidence_threshold=float(self.xgb_cfg["confidence_threshold"]),
        )
        action_value = int(actions[0])
        decision = ACTION_TO_DECISION.get(action_value, DEFAULT_FALLBACK_DECISION)
        confidence = float(confidences[0])

        rationale = generate_rationale(
            decision=decision,
            asset=payload.asset,
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
