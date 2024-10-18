# %% Imports
import pandas as pd
import numpy as np
import re
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')

import torch
from transformers import BertTokenizer, BertModel
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# %% Inputs

# What is your personal aesthetic? What colors, materials, and patterns describe your wardrobe or living spaces? (cottagecore, beach vibes, lumberjack, business formal, cozy, etc.)
aesthetic = "I like the small cozy feel of cottagecore, with the dark greens, wood burning stove, and homemade bread, reading a book on a rainy day. I also wear a lot of hawaiian shirts and like cool, breezy clothes. Overall, my aesthetic is comfy."

# What kind of weather do you like? (thunderstorms, low humidity, sunny afternoons, temperature)
weather = "I enjoy sunny, cloudless days in general, with the occasional rainy day. I like warm days and cool evenings."

# What biomes or geographical areas do you find yourself drawn to? (deserts, beaches, mountain tops, big cities, boreal forests, etc)
biome = "I like rainforests, dry temperate forests, islands, meadows, beaches, and mountains."

# What do you do for a living? (student, psychologist, retired chef, salary man)
living = "I am a data scientist."

# What is your dream job and why? (astronaut, stay-at-home parent, pro skater)
dream_job = "My dream job is to build sustainable, family-friendly living spaces. I appreciate the world around me, so I like seeing the rooftop gardens that cool down our cities, and I like seeing solar panels covering parking lots. I like seeing kids playing outside in parks and on the streets that are safe for them. I like seeing community gardens."

# What is your general disposition? (grumpy, jolly, content)
mood = "I am generally happy but not always."

# What are your hobbies? (hiking, exercising, video games, underwater basketweaving)
hobbies = "I like to code and work on data analysis projects. I like to hike and go to the gym to stay healthy."

# Lastly, a responsible pet owner knows their limits. How many pokemon do you expect to care for?
num_pokemon = 6

# %% Preprocessing
inputs = [aesthetic, weather, biome, living, dream_job, mood, hobbies]
inputs = [i.lower() for i in inputs]

# Input Bag of Words
inputs = list({aesthetic, weather, biome, living, dream_job, mood, hobbies})

# Vectorize inputes
vectorizer = CountVectorizer(stop_words='english')
vectored = vectorizer.fit_transform(inputs)
input_bow = vectored.toarray()
vocabulary = vectorizer.get_feature_names_out()
vocabulary = [text for text in vocabulary if not re.search(r'\d', text)]

# Get dex entries
full_dex = pd.read_csv("pokedex_full.csv")
vectorizer.fit(full_dex['Description'])
description_matrix = vectorizer.transform(full_dex['Description'])

features_df = pd.DataFrame(description_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Create a new column in full_dex that contains the unique features
full_dex['Features'] = features_df.apply(lambda row: ' '.join([word for word, val in zip(features_df.columns, row) if val > 0]), axis=1)

# %% Bag of Words
desc_vectored = vectorizer.fit_transform(full_dex['Features'])

# Transform user inputs into the same vocabulary
inp_vectored = vectorizer.transform(vocabulary)

# Vectorize into bag of words
word_count_matrix = np.vstack((desc_vectored.toarray(), inp_vectored.toarray()))

# Convert to numpy arrays to avoid TypeError
full_dex_bow = word_count_matrix[:len(full_dex)]  # Pokémon descriptions
input_bow = word_count_matrix[len(full_dex):]  # User inputs

# Compute cosine similarity
similarity = cosine_similarity(input_bow, full_dex_bow)

# Create a DataFrame for better visualization of results
similarity_df = pd.DataFrame(similarity, columns=full_dex['Pokemon'], index=vocabulary)

# Sum cosine similarities across all vocabulary terms (rows)
total_similarity = similarity_df.sum(axis=0)

# Sort Pokémon by similarity score in descending order and get the top 6
top_6_pokemon = pd.DataFrame(total_similarity.sort_values(ascending=False)).head(6)

# Display the top 6 Pokémon and their similarity scores
st.print("\n Bag of Words Top 6 Pokémon with highest similarity:")
st.print(top_6_pokemon)


# %% TF-IDF

# Combine the input text (user inputs) and Pokémon descriptions for consistent vectorization
inputs_combined = [' '.join(inputs)]  # User inputs combined into a single string
full_dex_descriptions = full_dex['Description'].tolist()

# Fit TF-IDF on both Pokémon descriptions and user inputs
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(full_dex_descriptions + inputs_combined)

# Separate TF-IDF matrices for Pokémon descriptions and the user input
full_dex_tfidf = tfidf_matrix[:len(full_dex_descriptions)]  # Pokémon descriptions
input_tfidf = tfidf_matrix[len(full_dex_descriptions):]  # User input (single vector)

# Convert to numpy arrays if needed
full_dex_tfidf = full_dex_tfidf.toarray()
input_tfidf = input_tfidf.toarray()

# Cosine Similarity: Now applied to TF-IDF vectors
cosine_sim_tfidf = cosine_similarity(input_tfidf, full_dex_tfidf)
cosine_sim_tfidf = cosine_sim_tfidf.flatten()

# Convert into a usable dataframe and sort by highest similarity
similarity_df_tfidf = pd.DataFrame({'Number': full_dex['Number'], 'Name': full_dex['Pokemon'], 'Type': full_dex['Type'], 'Generation': full_dex['Generation'], 'Similarity': cosine_sim_tfidf, 'Description': full_dex['Description']})

# Extract only the relevant part of the Description (after the " - ")
similarity_df_tfidf['Description'] = similarity_df_tfidf['Description'].str.split(' - ').str[-1]

# Sort by similarity and display top 'num_pokemon' Pokémon
similarity_df_tfidf = similarity_df_tfidf.sort_values(by='Similarity', ascending=False)

# Display the top matches
st.print("\n TF-IDF Top 6 Pokémon with highest similarity:")
st.print(similarity_df_tfidf.head(num_pokemon))

# %% GloVe

# Load GloVe embeddings from a file
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Function to tokenize text without lemmatization
def tokenize(text):
    return word_tokenize(text.lower())  # Tokenize and convert to lowercase

# Function to get the index of the token in the TF-IDF vocabulary
def get_tfidf_index(token, tfidf_vocab_dict):
    return tfidf_vocab_dict.get(token, None)

# Function to retrieve GloVe embedding for a given token
def get_glove_embedding(token, embeddings, embedding_dim=100):
    return embeddings.get(token, np.zeros(embedding_dim))  # Return the embedding if it exists, otherwise a zero vector

# Updated weighted_average_embedding function
def weighted_average_embedding(text, tfidf_vector, tfidf_vocab_dict, embeddings, embedding_dim=100):
    tokens = tokenize(text)
    embedding_sum = np.zeros(embedding_dim)
    total_weight = 0
    
    for token in tokens:
        idx = get_tfidf_index(token, tfidf_vocab_dict)
        if idx is not None:
            weight = tfidf_vector[0, idx]
            
            if weight > 0:
                glove_embedding = get_glove_embedding(token, embeddings, embedding_dim)
                
                if not np.all(glove_embedding == 0):
                    embedding_sum += glove_embedding * weight
                    total_weight += weight
    
    if total_weight > 0:
        return embedding_sum / total_weight
    else:
        return embedding_sum  # Return zero vector if no valid tokens

# Load GloVe embeddings
glove_file_path = 'Documents/GT/Potential Projects/glove.6B.100d.txt'
glove_embeddings = load_glove_embeddings(glove_file_path)

# Ensure the TF-IDF model (tfidf) is defined and fitted on your text data
# Example: tfidf = TfidfVectorizer().fit(your_text_data)

# Get the TF-IDF vocabulary as a dictionary (token: index)
tfidf_vocab_dict = tfidf.vocabulary_

# Calculate the weighted average embeddings for all Pokémon descriptions
dex_embeddings = []
for desc in full_dex['Features']:  # Make sure to replace 'Features' with the correct column name
    tfidf_vector = tfidf.transform([desc])
    dex_embedding = weighted_average_embedding(desc, tfidf_vector, tfidf_vocab_dict, glove_embeddings)
    dex_embeddings.append(dex_embedding)

# Calculate the weighted average embedding for user inputs
user_input = ' '.join(vocabulary)  # Combine inputs into one text
user_tfidf_vector = tfidf.transform([user_input])
user_embedding = weighted_average_embedding(user_input, user_tfidf_vector, tfidf_vocab_dict, glove_embeddings)

# Compute cosine similarity between user input embedding and dex embeddings
dex_embeddings = np.array(dex_embeddings)
user_embedding = user_embedding.reshape(1, -1)

cosine_sim = cosine_similarity(user_embedding, dex_embeddings)
cosine_sim = cosine_sim.flatten()

# Create a DataFrame to store the similarity results
similarity_df = pd.DataFrame({
    'Number': full_dex['Number'],
    'Name': full_dex['Pokemon'],
    'Type': full_dex['Type'],
    'Generation': full_dex['Generation'],
    'Similarity': cosine_sim,
    'Description': full_dex['Description']
})

# Sort by highest similarity and display top matches
similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
similarity_df['Description'] = similarity_df['Description'].str.split(' - ').str[-1]

# Display the top N results
st.print("\n GloVe Top 6 Pokémon with highest similarity:")
st.print(similarity_df.head(num_pokemon))

# %% BERT

# Function to get BERT embeddings for a sentence
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # We use the CLS token embedding (outputs.last_hidden_state[:, 0, :]) as the sentence embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

# Calculate BERT embeddings for Pokédex descriptions
dex_embeddings = []
for desc in full_dex_descriptions:
    embedding = get_bert_embedding(desc, tokenizer, model)
    dex_embeddings.append(embedding)

dex_embeddings = np.array(dex_embeddings).squeeze(axis=1)  # Convert to NumPy array

# Calculate BERT embedding for the user input
user_input = ' '.join(full_dex_descriptions)  # Combine inputs into one text
user_embedding = get_bert_embedding(user_input, tokenizer, model)
user_embedding = user_embedding.squeeze()

# Compute cosine similarity between user input embedding and dex embeddings
cosine_sim = cosine_similarity(user_embedding.reshape(1, -1), dex_embeddings)
cosine_sim = cosine_sim.flatten()

# Create a DataFrame to store the similarity results
similarity_df = pd.DataFrame({
    'Number': full_dex['Number'],
    'Name': full_dex['Pokemon'],
    'Type': full_dex['Type'],
    'Generation': full_dex['Generation'],
    'Similarity': cosine_sim,
    'Description': full_dex['Description']
})

# Sort by highest similarity and display top 6 matches
similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
similarity_df['Description'] = similarity_df['Description'].str.split(' - ').str[-1]

top_6_matches = similarity_df.head(num_pokemon)
st.print("\n BERT Top 6 Pokémon with highest similarity:")
st.print(top_6_matches)
