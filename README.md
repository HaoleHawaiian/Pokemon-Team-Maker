# Pokemon-Team-Maker
This project will give you a team of 6 pokemon based on inputs about your personality and lifestyle. Credit for the idea goes to the wonderful [Anna Zhu](https://www.linkedin.com/in/anna-zhu-r2d2/).

## Problem
How can we translate free form personality descriptions into meaningful matches with more structured data? Personality input is unstructured and possibly biased. Pokemon that have been around for 9 generations have different Pokedex entries per version per generation. 

## Approach
Originally, I split this between traditional vs new NLP methods, Term Frequency-Inverse Document Frequency (TF-IDF) and Bag of Words(BoW), looking at baseline and weighted word frequency, vs Global Vectors for Word Representation (GloVe) and Bidirectional Encoder Representations from Transformers (BERT). I did this to explore different levels of language understanding. For two years I let the app run, allowing users to vote on which team they felt best matched their input.

## API Data Pulls
This document shows how I connected to the Pokebase API and grabbed the pokedex entries and pokemon types. The results from this step are the pokedex_entries.csv and pokedex_types.csv.

## Data Preprocessing
Entries from the API had to be grouped and consolidated by Pokemon, the text cleaned and normalized, stopwords removed, and saved, to avoid calling the API every time someone runs the app. The result from this step is pokedex_full.csv. 

The main app will be using cosine similarities to compare Bag of Words and TF-IDF matrices to each other, but only the user inputs will change per use, so I want to preprocess the pokedex_full file ahead of time. I will only rerun this when the pokedex is updated, when a new game comes out.

## Pokemon Team Generator
This document is the completed proof of concept. Open this file to see how it works and download this file if you want to give it a try. It utilizes Bag of Words, TF-IDF, GloVe, and BERT models to compare the user input questions with the Pokedex entries.

## The App
Running on Streamlit locally, the GloVe and BERT models took several minutes, leading to a poor user experience, so for the current version of the app, I have removed those options. However, if you want to take the juptyer notebook and enter in your own inputs there, it runs much faster than streamlit, even with the GloVe and BERT options. Each pokemon recommendation includes a link that takes you to a page containing the Pokedex entries to compare to your inputs.

Since the project was originally released, I have added a way for users to vote on which team they feel best matches their inputs.

## Limitations
Pokedex descriptions are sparse, only one to two sentences. However, older pokemon have several generations and games with different variations of entries. There is of course, no ground truth that this can be compared to. To get around this, I have included the cosine similarity score of each Pokemon returned.
