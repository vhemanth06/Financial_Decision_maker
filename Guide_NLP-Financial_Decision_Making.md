# CLEF 2026 FinMMEval Lab – Task 3 Blueprint

Here is the exact, reality-grounded blueprint for tackling Task 3 of the CLEF 2026 FinMMEval Lab.

While algorithms designed for dynamic environments are excellent for complex state spaces, predicting asset prices on a rigidly constrained 233-step dataset requires a completely different paradigm. We must prioritize aggressive dimensionality reduction and signal stability over complex policy exploration.

This guide details the **Decoupled Late-Fusion Pipeline (FinBERT + XGBoost + Inference LLM)**. It is designed specifically to run flawlessly within a 16GB VRAM limit (Tesla T4 or P100) while maximizing your Sharpe Ratio.

---

## Phase 1: Ground-Truth Generation & Preprocessing

You cannot train a model without discrete targets. The dataset provides `future_price_diff` (a continuous float), but the competition requires a discrete action (BUY, SELL, HOLD).

### 1. Defining the Trading Threshold ($\tau$)

You must convert the continuous price difference into a supervised classification label. You cannot use a flat dollar amount because BTC and TSLA have vastly different volatilities. You must calculate a rolling standard deviation to define your threshold.

Let $D_t$ be the `future_price_diff` on day $t$. We define the target action $Y_t$ using a dynamic threshold $\tau$:

$$
Y_t =
\begin{cases}
1 \text{ (BUY)} & \text{if } D_t > \tau \\
-1 \text{ (SELL)} & \text{if } D_t < -\tau \\
0 \text{ (HOLD)} & \text{otherwise}
\end{cases}
$$

**Implementation Detail:** Set $\tau$ to represent a move large enough to overcome standard trading fees (e.g., a 0.5% move).

---

### 2. Taming the Unstructured Text (news, 10k, 10q)

Raw text will destroy your XGBoost model. We must convert paragraphs into numbers.

**The Tool:** Use `ProsusAI/finbert` from Hugging Face. It is incredibly lightweight and natively understands financial lexicon.

**The Execution:** Pass the daily news array through FinBERT. Extract the raw logits or the pooler output to generate a fixed 768-dimensional vector for every single day.

**Handling SEC Filings:** For TSLA's 10k and 10q data, write a simple Python script to truncate the text to the first 2,000 tokens (which usually contains the forward-looking management summary) before passing it to FinBERT.

---

### 3. Categorical Normalization

The dataset contains a momentum string. Map this to an integer:

- bullish = 1  
- bearish = -1  
- missing = 0  

---

## Phase 2: The Two-Stage Hybrid Architecture

This is the core engine. We separate the mathematical decision from the language generation to prevent hallucinations and guarantee low latency.

---

### Stage 2A: The Quantitative Brain (XGBoost)

You will build a tabular dataset where each row is a trading day.

**Features ($X$):**
- The 768 FinBERT dimensions  
- The mapped momentum integer  
- The raw prices (normalized as rolling percentages)  

**Target ($Y$):**
- Your engineered BUY(1), SELL(-1), HOLD(0) labels  

**The Model:** Train an `xgboost.XGBClassifier`.

**Why this wins:**  
XGBoost uses gradient boosted decision trees. It will rapidly identify which specific FinBERT embedding dimensions correlate with positive `future_price_diff`. It trains in seconds on a CPU, completely bypassing GPU bottlenecks. Set the `objective` parameter to `multi:softprob` so it outputs confidence percentages for each action.

---

### Stage 2B: The Qualitative Rationale (Lightweight LLM)

The competition requires a 50-word justification for the trade. We only trigger this LLM after XGBoost makes the mathematical decision.

**The Model:**  
Load `Qwen/Qwen2.5-1.5B-Instruct` or `meta-llama/Llama-3.2-1B-Instruct` via the `transformers` library in 4-bit quantization. This will consume roughly 2GB of VRAM.

**The Prompt Structure:**

- **System:**  
  "You are a concise financial analyst."

- **User:**  
  "Our quantitative system has signaled a [INSERT XGBOOST DECISION] for [ASSET]. The market momentum is [MOMENTUM]. Today's core news is: [INSERT FIRST 3 SENTENCES OF NEWS]. Write a strict 50-word rationale explaining this specific decision."

---

## Phase 3: Strict Validation Strategy

If you shuffle your dataset, your model will "see the future" and your out-of-sample metrics will collapse. Financial data must be evaluated chronologically.

---

### The Chronological Split (Based on the Aug 2025 - Mar 2026 data)

- **Train Set (First 60%):**  
  August 1, 2025 – December 15, 2025  

- **Validation Set (Next 20%):**  
  December 16, 2025 – January 31, 2026  
  *(Use this to tune XGBoost's learning rate and max depth)*

- **Test Set (Final 20%):**  
  February 1, 2026 – March 21, 2026  
  *(You only run this once to calculate your final submission metrics)*

---

### The Target Metrics

Do not optimize for pure accuracy. Optimize for the **Sharpe Ratio**.

$$
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
$$

- $R_p$: Your portfolio's return  
- $R_f$: The risk-free rate (assume 0 for daily intervals)  
- $\sigma_p$: The standard deviation of your daily returns  

**Strategy:**  
If XGBoost's output probabilities are low (e.g., 34% Buy, 33% Sell, 33% Hold), force the system to output HOLD. This prevents unnecessary trading, drops your volatility ($\sigma_p$), and mathematically drastically increases your Sharpe Ratio.

---

## Phase 4: Immediate Execution Steps

If you want to dominate this lab, follow this sequence exactly:

### 1. Deploy the "Dummy" API Today

The competition evaluates submissions via an API endpoint with a 3-minute timeout.

Write a FastAPI script that accepts the JSON payload, ignores the data, and returns:

```json
{"decision": "HOLD", "rationale": "Market volatility warrants a neutral position."}
```

Submit this immediately to ensure your networking and JSON parsing are compliant before adding ML.

---

### 2. Vectorize the Text

Write the loop to process the `news` column through FinBERT and save the output as a NumPy array (`.npy file`). You only want to run this expensive NLP process once, not every time you train XGBoost.

---

### 3. Train the Trees

Load the `.npy` file and the labels into a Pandas DataFrame and train your `XGBClassifier`.

---

Would you like me to write out the Python code for Step 2, focusing on the optimal way to batch the news text through FinBERT so it doesn't crash your Colab session?