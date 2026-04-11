"""DistilBERT sentence embeddings for runtime user queries."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_model_and_tokenizer(model_name: str = "distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """CLS token last hidden state as sentence embedding (same idea as original BERT notebook)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        # DistilBERT: first token = [CLS]
        cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        out_list.append(cls)
    return np.vstack(out_list)


def embed_single_text(text: str, tokenizer, model) -> np.ndarray:
    """Single query vector shape (hidden_size,)."""
    m = embed_texts([text], tokenizer, model, batch_size=1)
    return m[0]
