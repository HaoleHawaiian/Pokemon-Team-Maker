"""Sentence-Transformer embeddings (e.g. all-mpnet-base-v2) for semantic similarity."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 16,
    *,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """One row per text; L2-normalized vectors (cosine equals dot product)."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def embed_single(model: SentenceTransformer, text: str) -> np.ndarray:
    m = embed_texts(model, [text], batch_size=1)
    return m[0]
