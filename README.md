# Pokemon-Team-Maker

This project suggests a team of Pokémon based on free-form text about your personality and lifestyle. Credit for the idea goes to [Anna Zhu](https://www.linkedin.com/in/anna-zhu-r2d2/).

## Problem

How can we translate unstructured personality descriptions into meaningful matches with more structured data? Pokédex text is short and uneven across generations, and there is no single ground truth for a “correct” team—so each recommendation includes a cosine-similarity score.

## Approach

The Streamlit app compares two methods side by side:

- **Option 1 — Bag of words (BoW):** a sparse word-count style match (fast baseline).
- **Option 2 — DistilBERT:** contextual sentence embeddings (`distilbert-base-uncased`), with **precomputed** embeddings for every species so the app only runs the neural model on **your** text at click time.

Earlier experiments (TF‑IDF, GloVe, full BERT) lived in the archived notebooks under [`archive/notebooks/`](archive/notebooks/).

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app loads [`pokedex_full.csv`](Data/pokedex_full.csv) and matrix files from the local [`Data/`](Data/) folder when present; otherwise it downloads them from the `main` branch on GitHub (`Data/full_dex_bow.npy`, `Data/full_dex_distilbert.npy`, etc.). Ensure [`Data/full_dex_distilbert.npy`](Data/full_dex_distilbert.npy) is committed or available at that URL after you generate it (see below).

### Persistent voting (production)

Votes are stored in **PostgreSQL** when `DATABASE_URL` is set (recommended for Streamlit Cloud, Render, and similar hosts). Add the connection string to:

- **Streamlit Cloud:** app **Secrets** as `DATABASE_URL=postgresql://...`
- **Local:** environment variable `DATABASE_URL`, or `.streamlit/secrets.toml` (do not commit secrets)

If `DATABASE_URL` is not set, the app falls back to **`votes.db` SQLite** next to `streamlit_app.py` (fine for local development only).

**Vote semantics:** counts compare **Option 1 (BoW)** vs **Option 2 (DistilBERT)**. Older deployments that compared BoW vs TF‑IDF used the same option labels—treat historical totals accordingly or reset the `votes` table if you need a clean slate.

## Regenerating data (Python scripts)

Notebook workflows are replaced by scripts run from the repository root.

| Step | Command | Output |
|------|---------|--------|
| API pulls (slow; optional) | `pip install pokebase` then `python scripts/api_pull.py` | `Data/pokedex_entries.csv`, `Data/pokemon_types.csv` |
| Merge + BoW + TF‑IDF matrices | `python scripts/preprocess.py` | `Data/pokedex_full.csv`, `Data/full_dex_bow.npy`, `Data/full_dex_tfidf_sparse.npz` |
| DistilBERT embeddings | `python scripts/precompute_transformer_embeddings.py` | `Data/full_dex_distilbert.npy`, `Data/full_dex_distilbert_meta.json` |

After updating embeddings, push `full_dex_distilbert.npy` (and any updated CSV/npy assets) so hosted Streamlit can download them from `raw.githubusercontent.com`.

## Limitations

Pokédex blurbs are short; older species have many merged lines in one row. There is no objective “right” team—similarity scores are a transparency aid, not a verdict.

Under each recommended Pokémon, the app shows the **single clause** from that species’ merged Pokédex text with the strongest **word overlap** with your answers. That line is for readability; row-level **BoW / DistilBERT** scores still drive which species appear, not a separate score on each sentence.
