"""
Precompute DistilBERT embeddings for each row of pokedex_full.csv (CLS pooling).

Run from repo root:
  python scripts/precompute_transformer_embeddings.py

Outputs:
  Data/full_dex_distilbert.npy
  Data/full_dex_distilbert_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pokemon_team_maker.config import (  # noqa: E402
    DATA_DIR,
    DISTILBERT_META_FILENAME,
    DISTILBERT_MODEL_NAME,
    FULL_DEX_DISTILBERT_FILENAME,
)
from pokemon_team_maker.embeddings import embed_texts, load_model_and_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "pokedex_full.csv")
    parser.add_argument("--out", type=Path, default=DATA_DIR / FULL_DEX_DISTILBERT_FILENAME)
    parser.add_argument("--meta", type=Path, default=DATA_DIR / DISTILBERT_META_FILENAME)
    parser.add_argument("--model", type=str, default=DISTILBERT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")

    full_dex = pd.read_csv(csv_path)
    texts = full_dex["Description"].astype(str).tolist()

    tokenizer, model = load_model_and_tokenizer(args.model)
    arr = embed_texts(texts, tokenizer, model, batch_size=args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)

    h = hashlib.sha256()
    h.update(full_dex["Pokemon"].astype(str).str.cat(sep="|").encode())
    meta = {
        "model_name": args.model,
        "num_rows": int(len(full_dex)),
        "embedding_shape": list(arr.shape),
        "pokemon_column_sha256": h.hexdigest(),
    }
    args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} shape={arr.shape}")
    print(f"Wrote {args.meta}")


if __name__ == "__main__":
    main()
