# Pokemon-Team-Maker

This project suggests a team of Pokémon based on free-form text about your personality and lifestyle. Credit for the idea goes to [Anna Zhu](https://www.linkedin.com/in/anna-zhu-r2d2/).

## Problem

How can we translate unstructured personality descriptions into meaningful matches with more structured data? Pokédex text is short and uneven across generations, and there is no single ground truth for a “correct” team—so each recommendation includes a cosine-similarity score.

## Approach

The Streamlit app compares three methods side by side:

- **Option 1 — Bag of words (BoW):** a sparse word-count style match (fast baseline).
- **Option 2 — DistilBERT:** contextual embeddings (`distilbert-base-uncased`, CLS pooling), with **precomputed** vectors per species; at click time the model runs only on **your** text.
- **Option 3 — Sentence Transformers (MPNet):** `all-mpnet-base-v2`, tuned for semantic similarity; likewise **precomputed** per species with a single encode for the user profile.

Earlier experiments (TF‑IDF, GloVe, full BERT) lived in the archived notebooks under [`archive/notebooks/`](archive/notebooks/).

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app loads [`pokedex_full.csv`](Data/pokedex_full.csv) and matrix files from the local [`Data/`](Data/) folder when present; otherwise it downloads them from the `main` branch on GitHub (`Data/full_dex_bow.npy`, `Data/full_dex_distilbert.npy`, `Data/full_dex_mpnet.npy`, etc.). Ensure the precomputed `.npy` files are committed or available at those URLs after you generate them (see below).

### Persistent voting (production)

Votes are stored in **PostgreSQL** when `DATABASE_URL` is set (recommended for Streamlit Cloud, Render, and similar hosts). Add the connection string to:

- **Streamlit Cloud:** app **Secrets** as `DATABASE_URL=postgresql://...`
- **Local:** environment variable `DATABASE_URL`, or `.streamlit/secrets.toml` (do not commit secrets)

If `DATABASE_URL` is not set, the app falls back to **`votes.db` SQLite** next to `streamlit_app.py` (fine for local development only).

**Vote semantics:** counts compare **Option 1 (BoW)** vs **Option 2 (DistilBERT)** vs **Option 3 (MPNet)**. Older deployments with only two options used the same labels for the first two—treat historical totals accordingly, or reset the `votes` table (and ensure **Option 3** exists as a row) if you need a clean slate.

## Regenerating data (Python scripts)

Notebook workflows are replaced by scripts run from the repository root.

| Step | Command | Output |
|------|---------|--------|
| API pulls (slow; optional) | `pip install pokebase` then `python scripts/api_pull.py` | `Data/pokedex_entries.csv`, `Data/pokemon_types.csv` |
| Merge + BoW + TF‑IDF matrices | `python scripts/preprocess.py` | `Data/pokedex_full.csv`, `Data/full_dex_bow.npy`, `Data/full_dex_tfidf_sparse.npz` |
| DistilBERT embeddings | `python scripts/precompute_transformer_embeddings.py` | `Data/full_dex_distilbert.npy`, `Data/full_dex_distilbert_meta.json` |
| Sentence-Transformer (MPNet) embeddings | `python scripts/precompute_sentence_transformer_embeddings.py` | `Data/full_dex_mpnet.npy`, `Data/full_dex_mpnet_meta.json` |

After updating embeddings, push the `.npy` assets so hosted Streamlit can download them from `raw.githubusercontent.com`.

## Limitations

Pokédex blurbs are short; older species have many merged lines in one row. There is no objective “right” team—similarity scores are a transparency aid, not a verdict.

Under each recommended Pokémon, the app shows the **single clause** from that species’ merged Pokédex text with the strongest **word overlap** with your answers. That line is for readability; row-level **BoW / DistilBERT / MPNet** scores still drive which species appear, not a separate score on each sentence.
