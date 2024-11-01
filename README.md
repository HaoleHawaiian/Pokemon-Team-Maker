# Pokemon-Team-Maker
This project will give you a team of 6 pokemon based on inputs about your personality and lifestyle.

## API Data Pulls
This document shows how I connected to the Pokebase API and grabbed the pokedex entries and pokemon types. The results from this step are the pokedex_entries.csv and pokedex_types.csv

## Data Preprocessing
This document shows how I reformatted the API data pulls, from one line per pokedex entry and one line per pokemon type, to one line per pokemon, with all entries and types. The result from this step is pokedex_full.csv. 

The main app will be using cosine similarities to compare Bag of Words and TF-IDF matrices to each other, but only the user inputs will change per use, so I want to preprocess the pokedex_full file ahead of time. I will only rerun this when the pokedex is updated, same as the API pull.

## The App
The app is run through streamlit and may take a minute to wake up. When converting from the jupyter notebook explanations to streamlit, the GloVe and BERT models took several minutes, leading to a poorer user experience, so for the app right now, I have removed those options. However, if you want to take the juptyer notebook and enter in your own inputs there, it runs much faster than streamlit, even with the GloVe and BERT options. One next step is to allow users to select a team they think is a better fit for their personality. After a certain amount of collected data, the model will present "the team", based on user input on which is better, Bag of Words, or TF-IDF. Further next steps are to include a link on the pokemon's name and include a picture, generally presenting the information nicer than it is now.
