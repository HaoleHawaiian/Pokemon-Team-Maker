"""Team ranking from similarity scores (shared by BoW and transformer paths)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def format_pokemon_name(name: str) -> str:
    return name.replace(" ", "_")


def cosine_sim_bow_tfidf(input_vecs, full_dex_vecs, num_pokemon: int, dex_df: pd.DataFrame):
    """Sum cosine similarities across input rows vs dex (original app behavior)."""
    similarity = cosine_similarity(input_vecs, full_dex_vecs)
    similarity_df = pd.DataFrame(similarity, columns=dex_df["Pokemon"])
    total_similarity = similarity_df.sum(axis=0).sort_values(ascending=False)
    top_pokemon = total_similarity.head(num_pokemon).reset_index()
    top_pokemon.columns = ["Pokemon", "Similarity"]
    top_pokemon["Link"] = top_pokemon["Pokemon"].apply(
        lambda n: f"https://bulbapedia.bulbagarden.net/wiki/{format_pokemon_name(n)}_(Pokémon)"
    )
    return top_pokemon


def team_from_dense_similarity(
    user_embedding,
    dex_embeddings,
    num_pokemon: int,
    dex_df: pd.DataFrame,
) -> pd.DataFrame:
    """One user vector vs matrix of dex embeddings (one row per Pokemon)."""
    sim = cosine_similarity(user_embedding.reshape(1, -1), dex_embeddings).flatten()
    order = np.argsort(-sim)[:num_pokemon]
    top = pd.DataFrame(
        {
            "Pokemon": dex_df["Pokemon"].values[order],
            "Similarity": sim[order],
        }
    )
    top["Link"] = top["Pokemon"].apply(
        lambda n: f"https://bulbapedia.bulbagarden.net/wiki/{format_pokemon_name(n)}_(Pokémon)"
    )
    return top
