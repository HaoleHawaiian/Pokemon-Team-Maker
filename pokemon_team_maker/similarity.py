"""Team ranking from similarity scores (shared by BoW and transformer paths)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_STOP = frozenset(
    """
    a an and are as at be been being but by can could did do does doing done for from had has have
    he her hers him his how i if in into is it its just like me more most my no nor not of off on
    only or our ours out over own s same she should so some such t than that the their them then
    there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your
    """.split()
)


def format_pokemon_name(name: str) -> str:
    return name.replace(" ", "_")


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _iter_dex_fragments(dex_text: str) -> list[str]:
    """Split merged Pokédex blobs on form feeds, newlines, then sentence punctuation."""
    out: list[str] = []
    for block in re.split(r"[\f\n]+", dex_text):
        block = block.strip()
        if not block:
            continue
        for raw in re.split(r"[.;]+", block):
            p = raw.strip()
            if len(p) >= 18:
                out.append(p)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _trim_snippet(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def _strip_meta_preamble(line: str) -> str:
    """Remove `color habitat … - ` prefix used in merged pokedex_full descriptions."""
    line = line.strip()
    if " - " in line:
        return line.split(" - ", 1)[1].strip()
    return line


def best_matching_dex_snippet(
    user_profile: str,
    pokemon_name: str,
    dex_df: pd.DataFrame,
    max_chars: int = 280,
    desc_max_chars: int = 8000,
) -> str:
    """
    Clause from this species' merged Description with the most token overlap with user_profile.
    Row-level BoW/DistilBERT scores pick *which* species; this picks *which line* to show under the name.
    """
    rows = dex_df.loc[dex_df["Pokemon"] == pokemon_name, "Description"]
    if rows.empty:
        return ""
    dex_text = str(rows.iloc[0])[:desc_max_chars]
    fragments = _iter_dex_fragments(dex_text)
    if not fragments:
        raw = _strip_meta_preamble(dex_text) if dex_text else ""
        return _trim_snippet(raw, max_chars) if raw else ""

    utoks = _tokens(user_profile)
    if not utoks:
        raw = _strip_meta_preamble(max(fragments, key=len))
        return _trim_snippet(raw, max_chars)

    best_f = ""
    best_score = -1
    for f in fragments:
        f_narr = _strip_meta_preamble(f)
        if len(f_narr) < 12:
            continue
        score = len(_tokens(f_narr) & utoks)
        if score > best_score or (score == best_score and len(f_narr) > len(_strip_meta_preamble(best_f))):
            best_score = score
            best_f = f
    if best_score <= 0:
        best_f = max(fragments, key=len)
    raw = _strip_meta_preamble(best_f)
    return _trim_snippet(raw, max_chars)


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
