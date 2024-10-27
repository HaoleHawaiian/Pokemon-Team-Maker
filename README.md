# Pokemon-Team-Maker
This project will give you a team of 6 pokemon based on inputs about your personality and lifestyle.

## API Data Pulls
This document shows how I connected to the Pokebase API and grabbed the pokedex entries and pokemon types. The results from this step are the pokedex_entries.csv and pokedex_types.csv

## Data Preprocessing
This document shows how I reformatted the API data pulls, from one line per pokedex entry and one line per pokemon type, to one line per pokemon, with all entries and types. The result from this step is pokedex_full.csv. 

The main app will be using cosine similarities to compare Bag of Words and TF-IDF matrices to each other, but only the user inputs will change per use, so I want to preprocess the pokedex_full file ahead of time. I will only rerun this when the pokedex is updated, same as the API pull.
