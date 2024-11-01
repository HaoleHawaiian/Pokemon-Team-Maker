# Pokemon-Team-Maker
This project will give you a team of 6 pokemon based on inputs about your personality and lifestyle.

## API Data Pulls
This document shows how I connected to the Pokebase API and grabbed the pokedex entries and pokemon types. The results from this step are the pokedex_entries.csv and pokedex_types.csv

## Data Preprocessing
This document shows how I reformatted the API data pulls, from one line per pokedex entry and one line per pokemon type, to one line per pokemon, with all entries and types. The result from this step is pokedex_full.csv. 

The main app will be using cosine similarities to compare Bag of Words and TF-IDF matrices to each other, but only the user inputs will change per use, so I want to preprocess the pokedex_full file ahead of time. I will only rerun this when the pokedex is updated, same as the API pull.

## Pokemon Team Generator
This document is the completed proof of concept. Open this file to see how it works and download this file if you want to give it a try. It utilizes Bag of Words, TF-IDF, GloVe, and BERT models to compare the user input questions with the Pokedex entries.

## The App
I am working on getting the app up and running in Streamlit as the next step! Running on Streamlit locally, the GloVe and BERT models took several minutes, leading to a poorer user experience, so for the planned app, I have removed those options. However, if you want to take the juptyer notebook and enter in your own inputs there, it runs much faster than streamlit, even with the GloVe and BERT options. Further next steps are to allow users to select a team they think is a better fit for their personality and after gathering some data, will only present the crowd favorite. Additionally, I plan to include a link on the pokemon's name and include a picture, generally presenting the information nicer than it is now.
