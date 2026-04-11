"""
Build pokedex_full.csv, full_dex_bow.npy, and full_dex_tfidf_sparse.npz from raw API CSVs.

Run from repo root: python scripts/preprocess.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Allow running as script without installing package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_full_dex(dex_entries: pd.DataFrame, type_entries: pd.DataFrame) -> pd.DataFrame:
    dex_entries = dex_entries.copy()
    if "Habitat" in dex_entries.columns:
        dex_entries["Habitat"] = dex_entries["Habitat"].fillna("unknown")

    agg_descriptions = (
        dex_entries.groupby(["Pokemon", "Number", "Color", "Habitat", "Generation"], as_index=False)
        .agg({"Description": " ".join})
        .sort_values(by=["Number"])
        .reset_index(drop=True)
    )

    agg_types = type_entries.groupby(["Pokemon"])["Type"].agg(" ".join).reset_index()

    full_dex = pd.merge(agg_descriptions, agg_types, on="Pokemon", how="left")
    columns_order = ["Pokemon", "Number", "Color", "Habitat", "Type", "Generation", "Description"]
    full_dex = full_dex[columns_order]

    full_dex["Description"] = full_dex.apply(
        lambda row: (
            f"{row['Color']} {row['Habitat']} {row['Type']} - {row['Description']}"
            if row["Habitat"] != "unknown"
            else f"{row['Color']} {row['Type']} - {row['Description']}"
        ),
        axis=1,
    )
    full_dex["Description"] = full_dex["Description"].str.lower()
    return full_dex


def vectorize_and_save(full_dex: pd.DataFrame, data_dir: Path) -> None:
    vectorizer = CountVectorizer(stop_words="english")
    description_matrix = vectorizer.fit_transform(full_dex["Description"])
    features_df = pd.DataFrame(
        description_matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )
    full_dex = full_dex.copy()
    full_dex["Features"] = features_df.apply(
        lambda row: " ".join([word for word, val in zip(features_df.columns, row) if val > 0]),
        axis=1,
    )
    full_dex_bow = vectorizer.transform(full_dex["Features"]).toarray()
    np.save(data_dir / "full_dex_bow.npy", full_dex_bow)

    full_dex_descriptions = full_dex["Description"].tolist()
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(full_dex_descriptions)
    tfidf_sparse = sparse.csr_matrix(tfidf_matrix)
    sparse.save_npz(data_dir / "full_dex_tfidf_sparse.npz", tfidf_sparse)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Pokédex CSVs into full_dex and matrices.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "Data",
        help="Directory containing pokedex_entries.csv and pokemon_types.csv",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    entries_path = data_dir / "pokedex_entries.csv"
    types_path = data_dir / "pokemon_types.csv"
    if not entries_path.is_file():
        raise SystemExit(f"Missing {entries_path}")
    if not types_path.is_file():
        raise SystemExit(f"Missing {types_path}")

    dex_entries = pd.read_csv(entries_path)
    type_entries = pd.read_csv(types_path)

    full_dex = build_full_dex(dex_entries, type_entries)
    out_csv = data_dir / "pokedex_full.csv"
    full_dex.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} shape={full_dex.shape}")

    vectorize_and_save(full_dex, data_dir)
    print(f"Wrote {data_dir / 'full_dex_bow.npy'}")
    print(f"Wrote {data_dir / 'full_dex_tfidf_sparse.npz'}")


if __name__ == "__main__":
    main()
