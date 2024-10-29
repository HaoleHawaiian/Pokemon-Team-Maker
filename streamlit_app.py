import pandas as pd
import numpy as np
import io
import tempfile
# import re
import streamlit as st
import requests
from scipy.sparse import load_npz

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

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download file from {url}")

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
    
    return top_pokemon

def main():
    st.title("Pokemon Personality Team Generator")
    st.write("Input your preferences and see which Pokemon match your personality. Be as detailed as vague as you want, but more detail will give better results.")
    
    aesthetic = st.text_input("What is your personal aesthetic? What colors, materials, and patterns describe your wardrobe or living spaces? (cottagecore, beach vibes, lumberjack, business formal, cozy, etc.)", "I like the small cozy feel of cottages, with the dark greens, wood burning stove, and homemade bread, reading a book on a rainy day. I also wear a lot of hawaiian shirts and like cool, breezy clothes. Overall, my aesthetic is comfy.")
    weather = st.text_input("What kind of weather do you like? (thunderstorms, low humidity, sunny afternoons, temperature)", "I enjoy sunny, cloudless days in general, with the occasional rainy day. I like warm days and cool evenings.")
    biome = st.text_input("What biomes or geographical areas do you find yourself drawn to? (deserts, beaches, mountain tops, big cities, boreal forests, etc)", "I like rainforests, dry temperate forests, islands, meadows, beaches, and mountains.")
    living = st.text_input("What do you do for a living? (student, psychologist, retired chef, salary man)", "I am a data scientist.")
    dream_job = st.text_input("What is your dream job and why? (astronaut, stay-at-home parent, pro skater)", "My dream job is to build sustainable, family-friendly living spaces. I appreciate the world around me, so I like seeing the rooftop gardens that cool down our cities, and I like seeing solar panels covering parking lots. I like seeing kids playing outside in parks and on the streets that are safe for them. I like seeing community gardens.")
    mood = st.text_input("What is your general disposition? (grumpy, jolly, content)", "I am generally happy but not always.")
    hobbies = st.text_input("What are your hobbies? (hiking, exercising, video games, underwater basketweaving)", "I like to code and work on data analysis projects. I like to hike and go to the gym to stay healthy.")
    num_pokemon = st.number_input("Lastly, a responsible pet owner knows their limits. How many pokemon do you expect to care for?", value = 6, min_value = 1, max_value = 6)

    
    # Update these URLs with your actual GitHub raw URLs
    dex_bow_url = "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data/full_dex_bow.npy"
    dex_tf_idf_url = "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data/full_dex_tfidf_sparse.npz"
    full_dex_url = "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data/pokedex_full.csv"

    # Use a temporary file to save and load .npy and .npz files
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(download_file(dex_bow_url))
        temp_file_path = temp_file.name
    dex_bow_vec = np.load(temp_file_path)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(download_file(dex_tf_idf_url))
        temp_file_path = temp_file.name
    dex_tf_idf_vec = np.load_npz(temp_file_path)

    # Load the CSV file directly into a DataFrame
    full_dex = pd.read_csv(io.BytesIO(download_file(full_dex_url)))
    
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
        team = cosine_sim(input_bow, dex_bow_vec, num_pokemon, full_dex, vectorizer.get_feature_names_out())

        # TF-IDF
        input_tfidf = vectorize_inputs(inputs, tfidf_vectorizer)
        team_tfidf = cosine_sim(input_tfidf, dex_tf_idf_vec, num_pokemon, full_dex, tfidf_vectorizer.get_feature_names_out())
        
        # Display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Option 1:\n")
            st.dataframe(team)
            
        with col2:
            st.write("Option 2:\n")
            st.dataframe(team_tfidf)
            
        with col3:
            st.write("Option 3:\n")
            #st.dataframe()
if __name__ == "__main__":
    main()
