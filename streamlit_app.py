# %% Imports
import pandas as pd
import numpy as np
import re
import streamlit as st


st.title("Pokemon Personality Team Generator")


# %% Inputs

# What is your personal aesthetic? What colors, materials, and patterns describe your wardrobe or living spaces? (cottagecore, beach vibes, lumberjack, business formal, cozy, etc.)
aesthetic = st.text_input("What is your personal aesthetic? What colors, materials, and patterns describe your wardrobe or living spaces? (cottagecore, beach vibes, lumberjack, business formal, cozy, etc.)", "I like the small cozy feel of cottages, with the dark greens, wood burning stove, and homemade bread, reading a book on a rainy day. I also wear a lot of hawaiian shirts and like cool, breezy clothes. Overall, my aesthetic is comfy.")

# What kind of weather do you like? (thunderstorms, low humidity, sunny afternoons, temperature)
weather = st.text_input("What kind of weather do you like? (thunderstorms, low humidity, sunny afternoons, temperature)", "I enjoy sunny, cloudless days in general, with the occasional rainy day. I like warm days and cool evenings.")

# What biomes or geographical areas do you find yourself drawn to? (deserts, beaches, mountain tops, big cities, boreal forests, etc)
biome = st.text_input("What biomes or geographical areas do you find yourself drawn to? (deserts, beaches, mountain tops, big cities, boreal forests, etc)", "I like rainforests, dry temperate forests, islands, meadows, beaches, and mountains.")

# What do you do for a living? (student, psychologist, retired chef, salary man)
living = st.text_input("What do you do for a living? (student, psychologist, retired chef, salary man)", "I am a data scientist.")

# What is your dream job and why? (astronaut, stay-at-home parent, pro skater)
dream_job = st.text_input("What is your dream job and why? (astronaut, stay-at-home parent, pro skater)", "My dream job is to build sustainable, family-friendly living spaces. I appreciate the world around me, so I like seeing the rooftop gardens that cool down our cities, and I like seeing solar panels covering parking lots. I like seeing kids playing outside in parks and on the streets that are safe for them. I like seeing community gardens.")

# What is your general disposition? (grumpy, jolly, content)
mood = st.text_input("What is your general disposition? (grumpy, jolly, content)", "I am generally happy but not always.")

# What are your hobbies? (hiking, exercising, video games, underwater basketweaving)
hobbies = st.text_input("What are your hobbies? (hiking, exercising, video games, underwater basketweaving)", "I like to code and work on data analysis projects. I like to hike and go to the gym to stay healthy.")

# Lastly, a responsible pet owner knows their limits. How many pokemon do you expect to care for?
num_pokemon = st.number_input("Lastly, a responsible pet owner knows their limits. How many pokemon do you expect to care for?", 6, min_value = 1, max_value = 6)
