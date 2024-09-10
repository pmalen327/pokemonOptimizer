import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('Pokemon Database.csv')
effect_matrix = pd.read_csv('chart.csv')

# dropping values that do not contribute to battle stats
drop_cols = [
    'Pokemon Id', 'Classification', 'Alternate Form Name', 'Original Pokemon ID',
    'Legendary Type', 'Pokemon Height', 'Pokemon Weight', 'Primary Ability Description',
    'Secondary Ability', 'Secondary Ability Description', 'Hidden Ability',
    'Hidden Ability Description', 'Special Event Ability', 'Special Event Ability Description',
    'Male Ratio', 'Female Ratio', 'Base Happiness', 'Game(s) of Origin', 'Catch Rate', 'Experience Growth',
    'Experience Growth Total', 'Primary Egg Group', 'Secondary Egg Group', 'Egg Cycle Count',
    'Pre-Evolution Pokemon Id', 'Evolution Details'
]

df.drop(drop_cols, axis=1, inplace=True)
df.drop_duplicates(subset='Pokedex Number', keep='first', inplace=True)
types = effect_matrix.iloc[:, 0]
pokeB_choices = df.iloc[:, 1]


def calculate_type_advantage(type1_A, type2_A, type1_B, type2_B):
    score = 1.0
    if type1_A in effect_matrix and type1_B in effect_matrix[type1_A]:
        score *= effect_matrix[type1_A][type1_B]
    if type2_A != 'None' and type1_B in effect_matrix[type2_A]:
        score *= effect_matrix[type2_A][type1_B]
    if type1_A in effect_matrix and type2_B != 'None' and type2_B in effect_matrix[type1_A]:
        score *= effect_matrix[type1_A][type2_B]
    if type2_A != 'None' and type2_B != 'None' and type2_B in effect_matrix[type2_A]:
        score *= effect_matrix[type2_A][type2_B]
    return score

expanded_data = []
for index, row in df.iterrows():
    for b_choice in pokeB_choices:
        temp_row = row.copy()

        # REFACTOR THIS
        temp_row['Pokemon_B'] = b_choice
        temp_row['Type1_B'] = df[df['Pokemon_A'] == b_choice]['Type1_A'].values[0]
        temp_row['Type2_B'] = df[df['Pokemon_A'] == b_choice]['Type2_A'].values[0]
        temp_row['Advantage_Score'] = calculate_type_advantage(
            temp_row['Type1_A'], temp_row['Type2_A'], temp_row['Type1_B'], temp_row['Type2_B'])
        expanded_data.append(temp_row)

expanded_df = pd.DataFrame(expanded_data)

