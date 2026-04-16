
# CLEF-2026 Autonomous Trading Agent: AI Developer Instructions

## 1. System Context & Persona
You are an expert Quantitative Developer and ML Systems Architect. 
Your task is to build a modular, production-ready trading agent for the CLEF-2026 Financial Decision Making task (Task 3). 

**Core Constraints:**
- **Hardware:** Must run within a strict 16GB VRAM limit (Tesla T4).
- **Latency:** Inference must complete in under 3 minutes per API call.
- **Objective:** Optimize for the Sharpe Ratio (risk-adjusted returns), NOT pure accuracy.
- **Outputs:** Must return exactly one discrete action (BUY, HOLD, SELL) and a 50-word text rationale per daily JSON payload.
- **Architecture Paradigm:** Decoupled Late-Fusion (FinBERT -> PCA -> XGBoost -> Deterministic Rationale Rules). Do NOT attempt end-to-end deep learning fusion.
- **Rationale Constraint:** Do NOT use lightweight LLMs (Qwen/Llama) for rationale generation in this codebase.

---

## 2. Target Directory Structure
You must strictly adhere to the following file structure. Do not create files outside of this structure unless explicitly instructed.

```
clef2026_trading_agent/
├── data/                       # Ignored by Git
│   ├── raw/                    
│   ├── processed/              
│   └── embeddings/             
├── notebooks/                  
├── src/                        
│   ├── __init__.py
│   ├── data/                   
│   │   ├── __init__.py
│   │   ├── fetch_assets.py     
│   │   └── clean_data.py       
│   ├── features/               
│   │   ├── __init__.py
│   │   ├── labels.py           
│   │   └── text_encoder.py     
│   ├── models/                 
│   │   ├── __init__.py
│   │   ├── quant_xgboost.py    
│   │   └── rationale_rules.py   
│   ├── evaluation/             
│   │   ├── __init__.py
│   │   ├── metrics.py          
│   │   └── cross_val.py        
│   └── api/                    
│       ├── __init__.py
│       └── main.py             
├── configs/                    
│   └── config.yaml             
├── requirements.txt            
├── main_pipeline.py            
└── README.md                   
```

---

## 3. Implementation Phases & Task Breakdown
Execute the build in the following sequential phases. Do not move to a new phase until the current one is fully implemented and bug-free.

### Phase 1: Configuration & Environment
**Task 1.1:** Create `requirements.txt`. Include `pandas`, `numpy`, `torch`, `transformers`, `xgboost`, `scikit-learn`, `fastapi`, `uvicorn`, `pyyaml`, `tqdm`, `datasets`, `pyarrow`.
**Task 1.2:** Create `configs/config.yaml`. This file MUST contain all hyperparameters, file paths, model names (`ProsusAI/finbert`), threshold grids, and deterministic rationale-rule settings. **Rule: No hardcoded paths or parameters in `.py` files.**

### Phase 2: Data Ingestion (`src/data/`)
**Task 2.1:** Implement `fetch_assets.py`. Write a script to download the CLEF daily news parquet files (BTC, TSLA, MSFT, ETH, BMRN, MRNA) using Hugging Face datasets and save them to `data/raw/`.
**Task 2.2:** Implement `clean_data.py`. 
- Handle missing values gracefully (e.g., replace missing 10-K/10-Q arrays with an empty string).
- Map the categorical "momentum" column to integers: `bullish=1, bearish=-1, missing=0`.
- Output a single, consolidated chronological DataFrame to `data/processed/`.

### Phase 3: Feature Engineering (`src/features/`)
**Task 3.1:** Implement `labels.py`. 
- Calculate `future_price_diff` / `prices` to get daily returns.
- Write a function that uses a rolling standard deviation to determine a dynamic volatility threshold ($\tau$). 
- Generate discrete targets ($Y$): 1 if > $\tau$, -1 if < -$\tau$, 0 otherwise.
**Task 3.2:** Implement `text_encoder.py` (CRITICAL VRAM MANAGEMENT).
- Load `ProsusAI/finbert`. 
- Batch-process the `news` text to extract 768D pooler outputs. 
- **CRITICAL:** Train a `sklearn.decomposition.PCA` model to compress the 768D FinBERT vectors down to the top 30 principal components. Save the PCA model and the resulting `.npy` files to `data/embeddings/`.

### Phase 4: Modeling (`src/models/`)
**Task 4.1:** Implement `quant_xgboost.py`.
- Merge the tabular features (price, momentum) with the 30D PCA text embeddings.
- Train an `xgboost.XGBClassifier` with `objective="multi:softprob"`.
- Implement dynamic confidence thresholding: If the highest probability class is $< 0.45$, force the output to HOLD (0) to protect the Sharpe ratio.
**Task 4.2:** Implement `rationale_rules.py`.
- Build a deterministic rationale generator that accepts the XGBoost decision, confidence score, asset name, momentum, and day's text.
- Use config-driven keyword cues and confidence buckets to produce concise, risk-aware rationales.
- Constrain output generation to a strict `< 50 words` limit.

### Phase 5: Evaluation (`src/evaluation/`)
**Task 5.1:** Implement `cross_val.py`. Write a custom Combinatorial Purged Cross-Validation (CPCV) split to prevent data leakage from rolling features.
**Task 5.2:** Implement `metrics.py`. Write a custom evaluation function that calculates the **Sharpe Ratio**, assuming a standard 5 basis point (0.05%) transaction fee for every state change (e.g., changing from HOLD to BUY).

### Phase 6: Deployment & Orchestration
**Task 6.1:** Implement `api/main.py`. Build a FastAPI application with a `/predict` POST endpoint that receives the daily JSON schema.
- Must include an overarching `try/except` block that acts as a circuit breaker. If *anything* fails, return `{"decision": "HOLD", "rationale": "System fallback triggered due to data anomaly."}` to prevent API timeouts.
**Task 6.2:** Implement `main_pipeline.py`. This is the root orchestrator. It should import all functions from `src/` and run the entire training and embedding generation pipeline sequentially.

---

## 4. Coding Standards & Guardrails
- **Type Hinting:** Use strict Python type hints for all function arguments and return types.
- **Docstrings:** Use Google-style docstrings for every class and function. Explain *why* a function exists, not just what it does.
- **Modularity:** A function in `src/features/` should not import from `src/models/`. Data flows strictly left-to-right through the pipeline.
- **Logging:** Use Python's built-in `logging` library instead of `print()` statements. Configure logging at the top of `main_pipeline.py`.
- **Reproducibility:** Set a global random seed across `numpy`, `torch`, and `xgboost` in `config.yaml` to ensure deterministic results.
- **No LLM Runtime:** Do not introduce Qwen/Llama/quantized inference paths for rationale generation.

## 5. Maintenance & Handoff
- Update `README.md` with:
  1. Setup instructions (creating the virtual env, installing requirements).
  2. The exact bash command to run the end-to-end training pipeline.
  3. The exact bash command to start the FastAPI server.

