import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf


# dataset : https://www.kaggle.com/datasets/mrdew25/pokemon-database?select=Pokemon+Database.csv
# 151 Pokedex with duplicates, may exclude Mew
df = pd.read_csv('Pokemon Database.csv')

# dropping values that do not contribute to battle stats
drop_cols = [
    'Pokemon Id', 'Classification', 'Alternate Form Name', 'Original Pokemon ID',
    'Legendary Type', 'Pokemon Height', 'Pokemon Weight', 'Primary Ability Description',
    'Secondary Ability', 'Secondary Ability Description', 'Hidden Ability',
    'Hidden Ability Description', 'Special Event Ability', 'Special Event Ability Description',
    'Male Ratio', 'Female Ratio', 'Base Happiness', 'Game(s) of Origin', 'Catch Rate', 'Experience Growth',
    'Experience Growth Total', 'Primary Egg Group', 'Secondary Egg Group', 'Egg Cycle Count',
    'Pre-Evolution Pokemon Id'
]
df.drop(drop_cols, axis=1, inplace=True)
df = df.head(151)

# need to drop duplicate rows


# enumerate areas/gyms, this should be a cumulative list
# type matchup matrix, this will be a bitch



