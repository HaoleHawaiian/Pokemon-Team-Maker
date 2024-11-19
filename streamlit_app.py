import pandas as pd
import numpy as np
import io
import tempfile
# import re
import streamlit as st
import requests
from scipy.sparse import load_npz
import sqlite3

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import nltk
# from nltk.tokenize import word_tokenize
# nltk.download('wordnet')
# nltk.download('punkt')

# import torch
# from transformers import BertTokenizer, BertModel
# Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# Connect to an SQLite database to keep track of voting
conn = sqlite3.connect('votes.db')
cursor = conn.cursor()

# Create a table for storing votes if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS votes (
        option_name TEXT PRIMARY KEY,
        vote_count INTEGER DEFAULT 0
    )
''')

# Function to initialize the vote counts only if they are missing
def initialize_votes():
    cursor.execute('SELECT COUNT(*) FROM votes')
    count = cursor.fetchone()[0]
    
    if count == 0:
        cursor.execute('INSERT INTO votes (option_name, vote_count) VALUES ("Option 1", 0)')
        cursor.execute('INSERT INTO votes (option_name, vote_count) VALUES ("Option 2", 0)')
        conn.commit()

# Run initialization to ensure vote counts exist
initialize_votes()

# Function to get current vote counts
def get_votes(option_name):
    cursor.execute('SELECT vote_count FROM votes WHERE option_name = ?', (option_name,))
    result = cursor.fetchone()
    return result[0] if result else 0

# Function to update vote counts
def update_votes(option_name):
    current_votes = get_votes(option_name)
    cursor.execute('UPDATE votes SET vote_count = ? WHERE option_name = ?', (current_votes + 1, option_name))
    conn.commit()

def format_pokemon_name(name):
    return name.replace(" ", "_")

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download file from {url}")

# def load_glove_embeddings(url):
#     embeddings = {}
#     response = requests.get(url)
#     if response.status_code == 200:
#         for line in response.text.splitlines():
#             values = line.split()
#             word = values[0]
#             vector = np.array(values[1:], dtype='float32')
#             embeddings[word] = vector
#     else:
#         raise Exception(f"Failed to download GloVe embeddings from {url}")
    
#     return embeddings

def input_preprocess(aesthetic, weather, biome, living, dream_job, mood, hobbies):
    # Combine, lower, put into a list
    inputs = [aesthetic.lower(), weather.lower(), biome.lower(), living.lower(), dream_job.lower(), mood.lower(), hobbies.lower()]
    inputs = list(set(inputs))
    return inputs

def vectorize_inputs(inputs, vectorizer):
    return vectorizer.transform(inputs)

def cosine_sim(input_bow, full_dex_bow, num_pokemon, dex_df, feature_names=None):
    similarity = cosine_similarity(input_bow, full_dex_bow)
    
    # Create a DataFrame with Pokémon names as columns
    similarity_df = pd.DataFrame(similarity, columns=dex_df['Pokemon'])
    
    # Sum cosine similarities across all input terms
    total_similarity = similarity_df.sum(axis=0).sort_values(ascending=False)
    
    # Create a DataFrame for the top Pokémon, using Pokémon names
    top_pokemon = total_similarity.head(num_pokemon).reset_index()
    top_pokemon.columns = ['Pokemon', 'Similarity']
    
    # Create a column with the Bulbapedia links
    top_pokemon['Link'] = top_pokemon['Pokemon'].apply(lambda name: f"https://bulbapedia.bulbagarden.net/wiki/{format_pokemon_name(name)}_(Pokémon)")
    
    return top_pokemon

def display_team(team, column):
    # Display the team in the given column with two images and details in each row
    # Iterate through the team in steps of 2 (pairing Pokémon)
    for i in range(0, len(team), 2):
        # Create two columns for the pair of Pokémon
        col1, col2 = column.columns(2)
        
        # First Pokémon
        pokemon_1 = team.iloc[i]
        pokemon_name_1 = pokemon_1['Pokemon']
        similarity_1 = pokemon_1['Similarity']
        link_1 = pokemon_1['Link']
        formatted_name_1 = format_pokemon_name(pokemon_name_1).lower()
        image_url_1 = f"https://img.pokemondb.net/sprites/home/normal/{formatted_name_1}.png"

        with col1:
            st.image(image_url_1, caption=f"{pokemon_name_1} - Similarity: {similarity_1:.4f}", width=100)
            st.markdown(f"**[{pokemon_name_1}]({link_1})**")
            st.write(f"Similarity: {similarity_1:.4f}")
        
        # Check if there is a second Pokémon in the pair
        if i + 1 < len(team):
            pokemon_2 = team.iloc[i + 1]
            pokemon_name_2 = pokemon_2['Pokemon']
            similarity_2 = pokemon_2['Similarity']
            link_2 = pokemon_2['Link']
            formatted_name_2 = format_pokemon_name(pokemon_name_2).lower()
            image_url_2 = f"https://img.pokemondb.net/sprites/home/normal/{formatted_name_2}.png"

            with col2:
                st.image(image_url_2, caption=f"{pokemon_name_2} - Similarity: {similarity_2:.4f}", width=100)
                st.markdown(f"**[{pokemon_name_2}]({link_2})**")
                st.write(f"Similarity: {similarity_2:.4f}")

# def calculate_weighted_average_embeddings(descriptions, tfidf_vectorizer, glove_embeddings, embedding_dim=100):
#     tfidf_vocab_dict = tfidf_vectorizer.vocabulary_
#     dex_embeddings = []
    
#     for desc in descriptions:
#         tokens = word_tokenize(desc.lower())
#         embedding_sum = np.zeros(embedding_dim)
#         total_weight = 0
        
#         for token in tokens:
#             idx = tfidf_vocab_dict.get(token)
#             if idx is not None:
#                 weight = tfidf_vectorizer.transform([desc])[0, idx]
                
#                 if weight > 0:
#                     glove_embedding = glove_embeddings.get(token, np.zeros(embedding_dim))
                    
#                     if not np.all(glove_embedding == 0):
#                         embedding_sum += glove_embedding * weight
#                         total_weight += weight
        
#         dex_embeddings.append(embedding_sum / total_weight if total_weight > 0 else embedding_sum)
#     return np.array(dex_embeddings)

# # Function to compute similarities
# def compute_similarities(user_input, descriptions, tfidf_vectorizer, glove_embeddings):
#     user_input = ' '.join(user_input)
#     user_tokens = word_tokenize(user_input)
#     user_embedding = np.zeros(100)
#     total_weight = 0

#     for token in user_tokens:
#         idx = tfidf_vectorizer.vocabulary_.get(token)
#         if idx is not None:
#             weight = tfidf_vectorizer.transform([user_input])[0, idx]
#             glove_embedding = glove_embeddings.get(token, np.zeros(100))
#             user_embedding += glove_embedding * weight
#             total_weight += weight
            
#     user_embedding /= total_weight if total_weight > 0 else 1

#     # Calculate dex embeddings
#     dex_embeddings = calculate_weighted_average_embeddings(descriptions, tfidf_vectorizer, glove_embeddings)

#     # Compute cosine similarity
#     cosine_sim = cosine_similarity(user_embedding.reshape(1, -1), dex_embeddings).flatten()
#     return cosine_sim

def main():
    st.title("Pokemon Personality Team Generator")
    st.write("Input your preferences and see which Pokemon match your personality. Be as detailed or as vague as you want, but more information will give better results. This isn't like a Buzzfeed quiz where you'll pick from a dropdown list of answers, this is NLP!")
    
    # Initialize vote counts and voted flag in session state
    if "option_1_votes" not in st.session_state:
        st.session_state["option_1_votes"] = get_votes("Option 1")
    if "option_2_votes" not in st.session_state:
        st.session_state["option_2_votes"] = get_votes("Option 2")
    if "voted" not in st.session_state:
        st.session_state["voted"] = False
    if "team" not in st.session_state:
        st.session_state["team"] = None
    if "team_tfidf" not in st.session_state:
        st.session_state["team_tfidf"] = None
        
    aesthetic = st.text_area("What is your personal aesthetic? What colors, materials, and patterns describe your wardrobe or living spaces? (cottagecore, beach vibes, lumberjack, business formal, cozy, etc.)", placeholder = "I like the small cozy feel of cottages, with the dark greens, wood burning stove, and homemade bread, reading a book on a rainy day. I also wear a lot of Hawaiian shirts and like cool, breezy clothes. Overall, my aesthetic is comfy.")
    weather = st.text_area("What kind of weather do you like? (thunderstorms, low humidity, sunny afternoons, temperature)", placeholder = "I enjoy sunny, cloudless days in general, with the occasional rainy day. I like warm days and cool evenings.")
    biome = st.text_area("What biomes or geographical areas do you find yourself drawn to? (deserts, beaches, mountain tops, big cities, boreal forests, etc)", placeholder = "I like rainforests, dry temperate forests, islands, flowery, soft grassy meadows, tropical beaches, and mountains.")
    living = st.text_area("What do you do for a living? (student, psychologist, retired chef, salary man)", placeholder = "I am a data scientist. I work in predictive analytics to show the future in a way that can benefit those around me.")
    dream_job = st.text_area("What is your dream job and why? (astronaut, stay-at-home parent, pro skater)", placeholder = "My dream job is to build sustainable, family-friendly living spaces. I appreciate the world around me, so I like seeing the rooftop gardens that cool down our cities, and I like seeing solar panels covering parking lots. I like seeing kids playing outside in parks and on the streets that are safe for them. I like seeing community gardens.")
    mood = st.text_area("What is your general disposition? (grumpy, jolly, content)", placeholder = "I am generally happy but have trouble managing stress. Exercise and healthy diet help me manage and stay in the best mood.")
    hobbies = st.text_area("What are your hobbies? (hiking, exercising, video games, underwater basketweaving)", placeholder = "I like to code and work on goofy data analysis projects. I like to hike, surf, bake, playing the occasional video game, and go to the gym to stay healthy.")
    num_pokemon = st.number_input("Lastly, a responsible pet owner knows their limits. How many pokemon do you expect to care for?", value = 6, min_value = 1, max_value = 6)
    
    # Update these URLs with your actual GitHub raw URLs
    dex_bow_url = "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data/full_dex_bow.npy"
    dex_tf_idf_url = "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data/full_dex_tfidf_sparse.npz"
    full_dex_url = "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data/pokedex_full.csv"
    glove_url = "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data/glove.6B.100d.txt"
    
    # Use a temporary file to save and load .npy and .npz files
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(download_file(dex_bow_url))
        temp_file_path = temp_file.name
    dex_bow_vec = np.load(temp_file_path)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(download_file(dex_tf_idf_url))
        temp_file_path = temp_file.name
    dex_tf_idf_vec = load_npz(temp_file_path)

    # Load the CSV file directly into a DataFrame
    full_dex = pd.read_csv(io.BytesIO(download_file(full_dex_url)))

    # Load GloVe Embeddings
    # glove_embeddings = load_glove_embeddings(glove_url)
    
    # CountVectorizer setup (assuming consistent feature names)
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(full_dex['Description'])

    # TfIdfVectorizer setup for TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(full_dex['Description'])

    if st.button("Get My Team"):
        # Preprocess inputs
        inputs = input_preprocess(aesthetic, weather, biome, living, dream_job, mood, hobbies)
        
        # Bag of Words
        input_bow = vectorize_inputs(inputs, vectorizer)
        st.session_state["team"] = cosine_sim(input_bow, dex_bow_vec, num_pokemon, full_dex, vectorizer.get_feature_names_out())

        # TF-IDF
        input_tfidf = vectorize_inputs(inputs, tfidf_vectorizer)
        st.session_state["team_tfidf"] = cosine_sim(input_tfidf, dex_tf_idf_vec, num_pokemon, full_dex, tfidf_vectorizer.get_feature_names_out())
        
        #GloVe
        # similarity_results = compute_similarities(inputs, full_dex['Description'], tfidf_vectorizer, glove_embeddings)
        # similarity_df = pd.DataFrame({'Pokemon': full_dex['Pokemon'], 'Similarity': similarity_results})
        # similarity_df = similarity_df.sort_values(by='Similarity', ascending=False).head(num_pokemon)

        # Reset voting flag
        st.session_state["voted"] = False
        
        # Display
        col1, col2 = st.columns(2)
        
        if st.session_state["team"] is not None and st.session_state["team_tfidf"] is not None:
            st.write("This app is still new, and I am testing different methods of generating teams. Please vote for the team you feel best represents your personality below. Eventually, the best method will become the only method. Remember, don't vote for the team with Pokemon that you like more, vote for the one who's dex descriptions you feel best represent your inputs.")
            col1, col2 = st.columns(2)
    
            with col1:
                st.write("Option 1:")
                display_team(st.session_state["team"], col1)
                if st.button("Vote for Option 1") and not st.session_state["voted"]:
                    update_votes("Option 1")
                    st.session_state["option_1_votes"] += 1
                    st.session_state["voted"] = True
                    st.success("Thank you for voting for Option 1!")
    
            with col2:
                st.write("Option 2:")
                display_team(st.session_state["team_tfidf"], col2)
                if st.button("Vote for Option 2") and not st.session_state["voted"]:
                    update_votes("Option 2")
                    st.session_state["option_2_votes"] += 1
                    st.session_state["voted"] = True
                    st.success("Thank you for voting for Option 2!")
    
            # Show the current vote counts
            st.write(f"Option 1 Votes: {st.session_state['option_1_votes']}")
            st.write(f"Option 2 Votes: {st.session_state['option_2_votes']}")
            
        # with col3:
        #     st.write("Option 3:\n")
        #     st.dataframe(similarity_df)
            
        # with col4:
        #     st.write("Option 4:\n")
        #     #st.dataframe()

        # Footer with a link to your GitHub page
        st.markdown(
            """
            <div style="position: fixed; bottom: 0; width: 100%; text-align: center; background-color: #f1f1f1; padding: 10px;">
                <a href="https://github.com/HaoleHawaiian/Pokemon-Team-Maker" target="_blank">Visit my GitHub</a>
            </div>
            """, unsafe_allow_html=True
        )
if __name__ == "__main__":
    main()
