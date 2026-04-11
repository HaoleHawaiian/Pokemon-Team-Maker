import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from pokemon_team_maker.config import (
    DATA_DIR,
    DISTILBERT_MODEL_NAME,
    FULL_DEX_DISTILBERT_FILENAME,
    FULL_DEX_SENTENCE_TRANSFORMER_FILENAME,
    GITHUB_RAW_BASE,
    SENTENCE_TRANSFORMER_MODEL_NAME,
)
from pokemon_team_maker.embeddings import embed_single_text, load_model_and_tokenizer
from pokemon_team_maker.sentence_encoder import embed_single, load_model
from pokemon_team_maker.similarity import (
    best_matching_dex_snippet,
    cosine_sim_bow_tfidf,
    format_pokemon_name,
    team_from_dense_similarity,
)
from pokemon_team_maker.votes import OPTION_1, OPTION_2, OPTION_3, get_vote_store


@st.cache_data(ttl=3600, show_spinner=False)
def _download_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def _load_npy(name: str) -> np.ndarray:
    local = DATA_DIR / name
    if local.is_file():
        return np.load(local)
    url = f"{GITHUB_RAW_BASE}/{name}"
    return np.load(io.BytesIO(_download_bytes(url)), allow_pickle=False)


def _load_csv_bytes() -> bytes:
    local = DATA_DIR / "pokedex_full.csv"
    if local.is_file():
        return local.read_bytes()
    url = f"{GITHUB_RAW_BASE}/pokedex_full.csv"
    return _download_bytes(url)


@st.cache_data(ttl=3600, show_spinner=False)
def load_full_dex_df() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_load_csv_bytes()))


@st.cache_resource(show_spinner="Loading DistilBERT…")
def distilbert_model():
    return load_model_and_tokenizer(DISTILBERT_MODEL_NAME)


@st.cache_resource(show_spinner="Loading sentence model…")
def sentence_transformer_model():
    return load_model(SENTENCE_TRANSFORMER_MODEL_NAME)


def input_preprocess(aesthetic, weather, biome, living, dream_job, mood, hobbies):
    inputs = [
        aesthetic.lower(),
        weather.lower(),
        biome.lower(),
        living.lower(),
        dream_job.lower(),
        mood.lower(),
        hobbies.lower(),
    ]
    return list(set(inputs))


def vectorize_inputs(inputs, vectorizer):
    return vectorizer.transform(inputs)


def _option_heading(label: str) -> None:
    st.markdown(
        f'<p style="text-align:center;font-weight:600;font-size:1.05rem;margin:0 0 0.35rem 0;">{label}</p>',
        unsafe_allow_html=True,
    )


def _type_suffix_plain(full_dex: pd.DataFrame, pokemon_name: str) -> str:
    """` - Type1/Type2` for markdown after linked name; types are not linked."""
    rows = full_dex.loc[full_dex["Pokemon"] == pokemon_name, "Type"]
    if rows.empty:
        return ""
    t = str(rows.iloc[0]).strip()
    if not t or t.lower() == "nan":
        return ""
    parts = t.split()
    return " - " + "/".join(parts)


def display_team(team, column, user_text: str, full_dex: pd.DataFrame):
    """Sprites, links, and best lexical-overlap Pokédex clause; Similarity stays in `team` for ranking."""
    for i in range(0, len(team), 2):
        col1, col2 = column.columns(2)

        pokemon_1 = team.iloc[i]
        pokemon_name_1 = pokemon_1["Pokemon"]
        link_1 = pokemon_1["Link"]
        formatted_name_1 = format_pokemon_name(pokemon_name_1).lower()
        image_url_1 = f"https://img.pokemondb.net/sprites/home/normal/{formatted_name_1}.png"

        with col1:
            st.image(image_url_1, width=100)
            st.markdown(
                f"[**{pokemon_name_1}**]({link_1}){_type_suffix_plain(full_dex, pokemon_name_1)}"
            )
            if user_text.strip():
                snip1 = best_matching_dex_snippet(user_text, pokemon_name_1, full_dex)
                if snip1:
                    st.caption(snip1)

        if i + 1 < len(team):
            pokemon_2 = team.iloc[i + 1]
            pokemon_name_2 = pokemon_2["Pokemon"]
            link_2 = pokemon_2["Link"]
            formatted_name_2 = format_pokemon_name(pokemon_name_2).lower()
            image_url_2 = f"https://img.pokemondb.net/sprites/home/normal/{formatted_name_2}.png"

            with col2:
                st.image(image_url_2, width=100)
                st.markdown(
                    f"[**{pokemon_name_2}**]({link_2}){_type_suffix_plain(full_dex, pokemon_name_2)}"
                )
                if user_text.strip():
                    snip2 = best_matching_dex_snippet(user_text, pokemon_name_2, full_dex)
                    if snip2:
                        st.caption(snip2)


def main():
    st.set_page_config(
        page_title="Pokemon Personality Team Generator",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    vote_store = get_vote_store(Path(__file__).resolve().parent)
    vote_store.ensure_schema()

    st.title("Pokemon Personality Team Generator")
    st.write(
        "Input your preferences and see which Pokemon match your personality. Be as detailed or as vague as you want, "
        "but more information will give better results. This isn't like a Buzzfeed quiz where you'll pick from a dropdown "
        "list of answers, this is NLP!"
    )

    if "voted" not in st.session_state:
        st.session_state["voted"] = False
    if "team" not in st.session_state:
        st.session_state["team"] = None
    if "team_distilbert" not in st.session_state:
        st.session_state["team_distilbert"] = None
    if "team_mpnet" not in st.session_state:
        st.session_state["team_mpnet"] = None

    aesthetic = st.text_area(
        "What is your personal aesthetic? What colors, materials, and patterns describe your wardrobe or living spaces? (cottagecore, beach vibes, lumberjack, business formal, cozy, etc.)",
        placeholder="I like the small cozy feel of cottages, with the dark greens, wood burning stove, and homemade bread, reading a book on a rainy day. I also wear a lot of Hawaiian shirts and like cool, breezy clothes. Overall, my aesthetic is comfy.",
    )
    weather = st.text_area(
        "What kind of weather do you like? (thunderstorms, low humidity, sunny afternoons, temperature)",
        placeholder="I enjoy sunny, cloudless days in general, with the occasional rainy day. I like warm days and cool evenings.",
    )
    biome = st.text_area(
        "What biomes or geographical areas do you find yourself drawn to? (deserts, beaches, mountain tops, big cities, boreal forests, etc)",
        placeholder="I like rainforests, dry temperate forests, islands, flowery, soft grassy meadows, tropical beaches, and mountains.",
    )
    living = st.text_area(
        "What do you do for a living? (student, psychologist, retired chef, salary man)",
        placeholder="I am a data scientist. I work in predictive analytics to show the future in a way that can benefit those around me.",
    )
    dream_job = st.text_area(
        "What is your dream job and why? (astronaut, stay-at-home parent, pro skater)",
        placeholder="My dream job is to build sustainable, family-friendly living spaces. I appreciate the world around me, so I like seeing the rooftop gardens that cool down our cities, and I like seeing solar panels covering parking lots. I like seeing kids playing outside in parks and on the streets that are safe for them. I like seeing community gardens.",
    )
    mood = st.text_area(
        "What is your general disposition? (grumpy, jolly, content)",
        placeholder="I am generally happy but have trouble managing stress. Exercise and healthy diet help me manage and stay in the best mood.",
    )
    hobbies = st.text_area(
        "What are your hobbies? (hiking, exercising, video games, underwater basketweaving)",
        placeholder="I like to code and work on goofy data analysis projects. I like to hike, surf, bake, playing the occasional video game, and go to the gym to stay healthy.",
    )
    num_pokemon = st.number_input(
        "Lastly, a responsible pet owner knows their limits. How many pokemon do you expect to care for?",
        value=6,
        min_value=1,
        max_value=6,
    )

    full_dex = load_full_dex_df()

    dex_bow_vec = _load_npy("full_dex_bow.npy")

    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit(full_dex["Description"])

    tokenizer, d_model = distilbert_model()
    dex_distilbert = _load_npy(FULL_DEX_DISTILBERT_FILENAME)
    st_model = sentence_transformer_model()
    dex_mpnet = _load_npy(FULL_DEX_SENTENCE_TRANSFORMER_FILENAME)

    if st.button("Get My Team"):
        inputs = input_preprocess(
            aesthetic, weather, biome, living, dream_job, mood, hobbies
        )
        input_bow = vectorize_inputs(inputs, vectorizer)
        st.session_state["team"] = cosine_sim_bow_tfidf(
            input_bow, dex_bow_vec, num_pokemon, full_dex
        )

        user_text = " ".join(inputs)
        st.session_state["match_user_text"] = user_text
        with st.spinner("Finding matches…"):
            user_emb_db = embed_single_text(user_text, tokenizer, d_model)
            st.session_state["team_distilbert"] = team_from_dense_similarity(
                user_emb_db, dex_distilbert, num_pokemon, full_dex
            )
            user_emb_mp = embed_single(st_model, user_text)
            st.session_state["team_mpnet"] = team_from_dense_similarity(
                user_emb_mp, dex_mpnet, num_pokemon, full_dex
            )

        st.session_state["voted"] = False

    if (
        st.session_state["team"] is not None
        and st.session_state["team_distilbert"] is not None
        and st.session_state["team_mpnet"] is not None
    ):
        st.write(
            "This app is still new, and I am testing different methods of generating teams. "
            "Please vote for the team you feel best represents your personality below. Eventually, "
            "the best method will become the only method. Remember, don't vote for the team with "
            "Pokemon that you like more, vote for the one whose dex descriptions you feel best represent your inputs."
        )
        col1, col2, col3 = st.columns(3)

        match_text = st.session_state.get("match_user_text") or ""

        with col1:
            _option_heading("Option 1")
            opt1 = st.container(border=True)
            with opt1:
                display_team(st.session_state["team"], opt1, match_text, full_dex)
                if st.button("Vote for Option 1", key="vote_opt1") and not st.session_state["voted"]:
                    vote_store.increment(OPTION_1)
                    st.session_state["voted"] = True
                    st.success("Thank you for your vote!")
                    st.rerun()

        with col2:
            _option_heading("Option 2")
            opt2 = st.container(border=True)
            with opt2:
                display_team(st.session_state["team_distilbert"], opt2, match_text, full_dex)
                if st.button("Vote for Option 2", key="vote_opt2") and not st.session_state["voted"]:
                    vote_store.increment(OPTION_2)
                    st.session_state["voted"] = True
                    st.success("Thank you for your vote!")
                    st.rerun()

        with col3:
            _option_heading("Option 3")
            opt3 = st.container(border=True)
            with opt3:
                display_team(st.session_state["team_mpnet"], opt3, match_text, full_dex)
                if st.button("Vote for Option 3", key="vote_opt3") and not st.session_state["voted"]:
                    vote_store.increment(OPTION_3)
                    st.session_state["voted"] = True
                    st.success("Thank you for your vote!")
                    st.rerun()

    if st.session_state.get("voted"):
        option_1_votes = vote_store.get_count(OPTION_1)
        option_2_votes = vote_store.get_count(OPTION_2)
        option_3_votes = vote_store.get_count(OPTION_3)

        st.markdown("---")
        st.subheader("Current voting results")

        st.write(f"Option 1 votes: {option_1_votes}")
        st.write(f"Option 2 votes: {option_2_votes}")
        st.write(f"Option 3 votes: {option_3_votes}")

        vote_df = pd.DataFrame(
            {
                "Option": ["Option 1", "Option 2", "Option 3"],
                "Votes": [option_1_votes, option_2_votes, option_3_votes],
            }
        )
        st.bar_chart(vote_df.set_index("Option"))

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; width: 100%; text-align: center; background-color: #f1f1f1; padding: 10px;">
            <a href="https://github.com/HaoleHawaiian/Pokemon-Team-Maker" target="_blank">Visit my GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
