# Financial Decision Maker

A comprehensive financial decision-making pipeline that combines news sentiment analysis with financial data to predict asset price movements. This project leverages FinBERT embeddings, asset-specific threshold tuning, and chronological data splitting to build a robust machine learning dataset for financial forecasting.

## Project Overview

This project consolidates financial data preparation into a unified pipeline that:
- Loads multi-asset financial news and filing data from the TheFinAI dataset
- Performs robust preprocessing and asset-specific label tuning
- Extracts contextual embeddings using FinBERT (fine-tuned BERT for finance)
- Creates structured records with historical price context
- Generates balanced train/validation/test splits without temporal leakage

## Features

### Multi-Asset Coverage
Supports the following financial instruments:
- **Cryptocurrencies**: BTC (Bitcoin), ETH (Ethereum)
- **Tech Stocks**: TSLA (Tesla), MSFT (Microsoft)
- **Biotech Stocks**: BMRN (Biomarin), MRNA (Moderna)

### Data Sources
- **News Articles**: Daily financial news from TheFinAI dataset
- **Regulatory Filings**: 10-K and 10-Q documents
- **Market Data**: Historical prices and returns
- **Sentiment Indicators**: Pre-computed momentum signals (bullish/bearish)

### Intelligent Labeling Strategy
- **Asset-Specific Thresholds**: Each asset gets its own threshold of uncertainty (tou) tuned to achieve balanced label distributions
- **Balanced Labels**: Targets equal distribution of negative (-1), neutral (0), and positive (1) labels
- **Quantile-Based Optimization**: Searches across quantile ranges (5%-45%) to find optimal price movement thresholds

### Advanced Text Processing
- **FinBERT Embeddings**: Extracts 768-dimensional contextual embeddings from financial text
- **Multi-Source Text Aggregation**: Combines news articles with regulatory filing text
- **Batch Processing**: GPU-accelerated embedding extraction with configurable batch sizes

### Temporal Integrity
- **Chronological Splits**: Maintains temporal order (70% train, 15% validation, 15% test)
- **Per-Asset Splitting**: Ensures no temporal leakage within individual assets
- **Historical Context**: Preserves price history for each record (configurable lookback window)

### Flexible Output Formats
Two complementary output formats for different use cases:
1. **Tabular DataFrames** (`train_df`, `val_df`, `test_df`): Quick inspection, debugging, and direct ML pipeline integration
2. **JSONL Records**: Nested structured data for sophisticated ML frameworks and APIs

## Installation

### Requirements
```bash
Python >= 3.8
torch >= 2.0
transformers >= 4.30
pandas >= 1.5
numpy >= 1.23
huggingface-hub >= 0.16
```

### Setup
```bash
pip install torch transformers pandas numpy huggingface-hub
```

### Hugging Face Authentication
For optimal dataset and model downloads, set your Hugging Face token:
```bash
export HF_TOKEN="your_hf_token_here"
```

Or configure it in the notebook directly.

## Usage

### Basic Pipeline

Run the preprocessing pipeline notebook:
```bash
jupyter notebook 01-phase1-preprocessing-pipeline.ipynb
```

The notebook will:
1. Load data for all 6 assets from TheFinAI dataset
2. Calculate returns based on daily price changes
3. Tune asset-specific thresholds using quantile optimization
4. Generate balanced three-class labels
5. Extract FinBERT embeddings (requires GPU for reasonable speed)
6. Create formatted records with metadata
7. Split data chronologically per asset
8. Export train/val/test splits as both DataFrames and JSONL

### Output Files

- **train.jsonl**: Training set records (≈70% of data per asset)
- **val.jsonl**: Validation set records (≈15% of data per asset)
- **test.jsonl**: Test set records (≈15% of data per asset)

### Record Structure

Each JSONL record contains:
```json
{
  "date": "2024-01-15",
  "symbol": ["TSLA"],
  "price": {"TSLA": 185.50},
  "history_price": {
    "TSLA": [
      {"date": "2024-01-14", "price": 184.20},
      {"date": "2024-01-13", "price": 183.80}
    ]
  },
  "news": {"TSLA": ["Article 1", "Article 2"]},
  "10k": {"TSLA": ["Filing content"]},
  "10q": {"TSLA": ["Quarterly filings"]},
  "momentum": {"TSLA": "bullish"},
  "momentum_encoded": 1,
  "finbert_vector": [0.124, -0.056, ...],
  "label": 1
}
```

## Pipeline Details

### 1. Data Loading
- Loads parquet files from TheFinAI's Hugging Face dataset
- Combines data from 6 assets into unified DataFrame
- Typical dataset size: ~5,000+ rows per asset

### 2. Returns Calculation
```
returns = (future_price_diff) / (current_price)
```

### 3. Asset-Specific Threshold Tuning
For each asset:
- Tests quantiles from 5% to 45% of absolute returns
- For each candidate threshold, creates labels:
  - **+1**: if returns > threshold
  - **-1**: if returns < -threshold
  - **0**: otherwise
- Selects threshold that minimizes distance to balanced (1/3, 1/3, 1/3) distribution

### 4. Text Preprocessing
- **News Aggregation**: Multiple articles combined with [SEP] separator
- **Filing Aggregation**: 10-K and 10-Q documents concatenated
- **Combined Text**: All sources merged with proper spacing
- **Truncation**: Capped at 512 tokens for FinBERT

### 5. FinBERT Embedding
- Model: `ProsusAI/finbert` (pre-trained on financial documents)
- Output: 768-dimensional embeddings from pooled output
- Batch processing: Configurable batch size (default: 16)
- Device: Automatic GPU detection with CPU fallback

### 6. Chronological Splitting
```
Per asset:
- Train:  0-70% (earliest to recent)
- Val:   70-85% (recent history)
- Test:  85-100% (most recent data)
```

This prevents lookahead bias and realistic time-series evaluation.

## Data Exploration

The `data-loader.ipynb` notebook provides:
- Initial data loading and exploration
- Returns calculation and validation
- Threshold tuning demonstration
- Label balance visualization
- Data quality checks

Run this first to understand your data before using the full pipeline.

## Example Workflow

```python
# Load the preprocessed data
import json
import pandas as pd

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# Load splits
train_data = load_jsonl('train.jsonl')
val_data = load_jsonl('val.jsonl')
test_data = load_jsonl('test.jsonl')

# Convert to DataFrames for quick analysis
train_df = pd.DataFrame(train_data)
print(f"Training set: {len(train_df)} records")
print(f"Label distribution:\n{train_df['label'].value_counts()}")

# Access embeddings for your model
embeddings = [record['finbert_vector'] for record in train_data]
labels = [record['label'] for record in train_data]
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantile_grid` | 5%-45% | Range of quantiles to test for threshold tuning |
| `batch_size` | 16 | Batch size for FinBERT embedding extraction |
| `max_length` | 512 | Max tokens for BERT tokenization |
| `train_pct` | 70% | Percentage of data for training |
| `val_pct` | 15% | Percentage of data for validation |
| `LOOKBACK_DAYS` | ∞ | Historical price lookback window |

## Performance Notes

- **GPU Acceleration**: FinBERT embedding extraction is significantly faster with CUDA (10-50x speedup)
- **Memory**: With batch_size=16, requires ~8GB GPU memory
- **Processing Time**: Full pipeline with 30,000 records: ~5-15 minutes (GPU), ~1-2 hours (CPU)

## Project Structure

```
Financial_Decision_maker/
├── 01-phase1-preprocessing-pipeline.ipynb  # Main preprocessing pipeline
├── data-loader.ipynb                        # Data exploration and loading
├── README.md                                # This file
├── train.jsonl                              # Generated: training data
├── val.jsonl                                # Generated: validation data
└── test.jsonl                               # Generated: test data
```

## Next Steps

1. **Model Training**: Use the generated JSONL files with your favorite ML framework
2. **Embedding Analysis**: Explore FinBERT representations via dimensionality reduction
3. **Backtesting**: Validate model predictions against realized returns
4. **Custom Thresholds**: Adjust `quantile_grid` or asset-specific `tou` parameters
5. **Additional Features**: Add technical indicators, volatility metrics, or sentiment scores

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in FinBERT embedding function (e.g., 8 instead of 16)
- Process assets sequentially instead of all at once

### Missing Data or NaN Values
- Check raw parquet files: `df.isnull().sum()`
- Verify asset names match exactly (case-sensitive)
- Run data quality checks in `data-loader.ipynb`

### Slow Embedding Extraction
- Verify GPU availability: `torch.cuda.is_available()`
- Check GPU memory: `nvidia-smi`
- Consider using pre-computed embeddings or cache

## Performance Insights

- **Label Balance**: Tuning per asset achieves ~33% distribution for -1, 0, 1 labels
- **Embedding Quality**: FinBERT captures financial sentiment and domain-specific patterns
- **Temporal Split**: Realistic 70-15-15 split prevents lookahead bias
- **Data Redundancy**: Some assets have overrepresented dates; JSONL format handles this naturally

## References

- **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **Dataset**: [TheFinAI/daily_news](https://huggingface.co/datasets/TheFinAI/daily_news)
- **Documentation**: Transformer models, PyTorch, Hugging Face

## License

Refer to the [LICENSE](./LICENSE) file in the repository for licensing information.

## Contact

For questions or issues, please reach out to the project maintainers.