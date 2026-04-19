# CLEF-2026 Autonomous Trading Agent

Modular late-fusion trading system for CLEF-2026 Financial Decision Making Task 3.

Architecture:
- FinBERT text encoding (`ProsusAI/finbert`)
- PCA compression to 30D
- XGBoost multiclass decision model
- Deterministic rationale rules engine (no Qwen/Llama/quantized LLM runtime)

## 1. Setup

```bash
cd Financial_Decision_maker
conda activate nlp
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run End-to-End Training Pipeline

If the CLEF dataset is private, set your Hugging Face token first:

```bash
export HF_TOKEN=your_huggingface_token
```

```bash
cd Financial_Decision_maker
python main_pipeline.py --config configs/config.yaml
```

What this runs in sequence:
1. Download raw asset datasets from Hugging Face.
2. Clean and consolidate time-series data.
3. Generate dynamic volatility-threshold labels.
4. Build FinBERT embeddings and 30D PCA projections.
5. Run CPCV Sharpe evaluation with transaction costs.
6. Calibrate confidence-threshold policy from CPCV.
7. Train and save final XGBoost model.

## 3. Start FastAPI Inference Server

```bash
cd Financial_Decision_maker
CLEF_CONFIG_PATH=configs/config.yaml uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## 4. API Contract

### POST `/predict`

Example request (nested per-asset format):

```json
{
  "date": "2024-10-10",
  "price": {
    "BTC": 64500.0
  },
  "symbol": {
    "BTC": "BTC"
  },
  "momentum": {
    "BTC": "bullish"
  },
  "news": {
    "BTC": [
      "Bitcoin rises as ETF inflows continue."
    ]
  },
  "10k": {
    "BTC": null
  },
  "10q": {
    "BTC": null
  },
  "history_price": {
    "BTC": [
      {"date": "2024-10-08", "price": 63100.0},
      {"date": "2024-10-09", "price": 63850.0}
    ]
  }
}
```

Legacy flat payloads are still accepted for backward compatibility:

```json
{
  "asset": "BTC",
  "prices": 64500.0,
  "momentum": "bullish",
  "news": "Bitcoin rises as ETF inflows continue.",
  "filing_10k": "",
  "filing_10q": ""
}
```

Example response:

```json
{
  "decision": "BUY",
  "rationale": "Momentum and sentiment support upside, but volatility remains elevated and reversal risk persists if macro data surprises."
}
```

Circuit-breaker behavior:
- Any runtime failure returns:
  - `{"decision": "HOLD", "rationale": "System fallback triggered due to data anomaly."}`

## 5. Deterministic Rationale Generation (LLM-Free)

- Rationale text is built from transparent rules using:
  - XGBoost decision class (BUY/HOLD/SELL)
  - XGBoost confidence score
  - Momentum signal
  - Lightweight keyword balance from the day's text context
- API-side rolling price history is used to compute RSI/MA-ratio/volatility features
  so live inference better matches training feature construction.
- Output is always hard-capped to 50 words.
- This removes all lightweight LLM runtime dependencies from the inference path.

## 6. Deploy on Render (Free Tier)

This repository now includes [render.yaml](render.yaml) for one-click Render deployment.

### Option A: Blueprint Deploy (recommended)

1. Push this repo to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Connect your repo and deploy. Render will read [render.yaml](render.yaml).

### Option B: Manual Web Service

Use these settings in Render:

- Runtime: Python
- Plan: Free
- Build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

- Start command:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

- Environment variables:
  - `CLEF_CONFIG_PATH=configs/config.yaml`
  - `HF_TOKEN=<your_token>` (only if private model/dataset access is needed)

### Important notes for this project

- Render requires binding to `$PORT`; do not hardcode `8000` in start command.
- Free tier services sleep when idle and take time to wake up.
- Ensure model artifacts referenced by `configs/config.yaml` are present in the repo at deploy time, especially `data/embeddings/finbert_pca_model.pkl`, `data/processed/xgb_model.json`, and `data/processed/xgb_policy.json`.
