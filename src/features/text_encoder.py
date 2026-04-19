from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils import load_config
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def _resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device from config while supporting auto mode.

    Args:
        device_name: Requested device string from config.

    Returns:
        A torch.device instance.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def encode_texts_with_finbert(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device_name: str,
    use_fp16_on_cuda: bool,
) -> np.ndarray:
    """Encode text into FinBERT dense vectors using memory-aware batched inference.

    Args:
        texts: Iterable of text snippets to encode.
        model_name: Hugging Face model name.
        batch_size: Number of samples per inference batch.
        max_length: Tokenization truncation length.
        device_name: Device string from config.
        use_fp16_on_cuda: Whether to switch model to FP16 on CUDA for lower VRAM use.

    Returns:
        Matrix of sentence embeddings with shape (n_samples, 768).
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    device = _resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Force safetensors to avoid torch.load security restriction on torch<2.6.
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model = model.to(device)

    if device.type == "cuda" and use_fp16_on_cuda:
        model = model.half()

    model.eval()

    embeddings: list[np.ndarray] = []
    for start_index in tqdm(range(0, len(texts), batch_size), desc="FinBERT encoding", leave=False):
        batch_texts = [str(item) for item in texts[start_index : start_index + batch_size]]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            if getattr(outputs, "pooler_output", None) is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]

        embeddings.append(pooled.detach().float().cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


def fit_pca_and_save(
    embeddings: np.ndarray,
    n_components: int,
    seed: int,
    pca_model_path: Path,
    reduced_embeddings_path: Path,
) -> np.ndarray:
    """Fit PCA on FinBERT vectors and persist both model and reduced matrix.

    Args:
        embeddings: Full FinBERT embedding matrix.
        n_components: Requested number of principal components.
        seed: Random seed for reproducibility.
        pca_model_path: Output path for serialized PCA model.
        reduced_embeddings_path: Output path for reduced embedding matrix.

    Returns:
        PCA-compressed embedding matrix.

    Raises:
        ValueError: If the embedding matrix is empty.
    """
    if embeddings.size == 0:
        raise ValueError("Cannot fit PCA on an empty embedding matrix.")

    effective_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    if effective_components != n_components:
        LOGGER.warning(
            "Requested %d PCA components but using %d due to matrix shape %s",
            n_components,
            effective_components,
            embeddings.shape,
        )

    pca_model = PCA(n_components=effective_components, random_state=seed)
    reduced_embeddings = pca_model.fit_transform(embeddings).astype(np.float32)

    pca_model_path.parent.mkdir(parents=True, exist_ok=True)
    with pca_model_path.open("wb") as file_obj:
        pickle.dump(pca_model, file_obj)

    np.save(reduced_embeddings_path, reduced_embeddings)
    return reduced_embeddings


def build_finbert_pca_embeddings(config: dict[str, Any]) -> np.ndarray:
    """Generate and persist FinBERT and PCA embeddings from labeled data.

    Args:
        config: Full project configuration dictionary.

    Returns:
        PCA-compressed embedding matrix.
    """
    paths_cfg = config["paths"]
    features_cfg = config["features"]
    finbert_cfg = config["finbert"]

    input_path = Path(paths_cfg["labeled_data"])
    full_embeddings_path = Path(paths_cfg["finbert_embeddings"])
    reduced_embeddings_path = Path(paths_cfg["finbert_pca_embeddings"])
    pca_model_path = Path(paths_cfg["pca_model"])

    frame = pd.read_parquet(input_path)
    text_column = features_cfg["text_column"]
    texts = frame[text_column].fillna("").astype(str).tolist()

    full_embeddings = encode_texts_with_finbert(
        texts=texts,
        model_name=finbert_cfg["model_name"],
        batch_size=finbert_cfg["batch_size"],
        max_length=finbert_cfg["max_length"],
        device_name=finbert_cfg["device"],
        use_fp16_on_cuda=finbert_cfg["use_fp16_on_cuda"],
    )

    full_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(full_embeddings_path, full_embeddings)

    reduced_embeddings = fit_pca_and_save(
        embeddings=full_embeddings,
        n_components=finbert_cfg["pca_components"],
        seed=config["seed"]["numpy"],
        pca_model_path=pca_model_path,
        reduced_embeddings_path=reduced_embeddings_path,
    )

    LOGGER.info(
        "Saved FinBERT embeddings: full=%s reduced=%s pca=%s",
        full_embeddings_path,
        reduced_embeddings_path,
        pca_model_path,
    )
    return reduced_embeddings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Generate FinBERT embeddings and PCA projections.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """Run text encoding and PCA projection as a standalone script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    build_finbert_pca_embeddings(config)


if __name__ == "__main__":
    main()
