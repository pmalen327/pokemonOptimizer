import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = (
    "Cuda GPU"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "CPU"
)

print(f"Using {device}")

df = pd.read_csv('Pokemon Database.csv')
effect_matrix = pd.read_csv('chart.csv')

# dropping values that do not contribute to base stats
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
df.fillna('none', inplace=True)


def calculate_type_advantage(type1_A, type2_A, type1_B, type2_B):
    # ensure all types are cleaned and lowercased
    type1_A = type1_A.strip().lower()
    type2_A = type2_A.strip().lower()
    type1_B = type1_B.strip().lower()
    type2_B = type2_B.strip().lower()

    score = 1.0
    
    # compare Type1_A with both Type1_B and Type2_B
    if type1_A in types:
        if type1_B in types[type1_A]:
            score *= types[type1_A][type1_B]
        if type2_B != 'none' and type2_B in types[type1_A]:
            score *= types[type1_A][type2_B]
    
    # compare Type2_A with both Type1_B and Type2_B if Type2_A is not 'none'
    if type2_A != 'none' and type2_A in types:
        if type1_B in types[type2_A]:
            score *= types[type2_A][type1_B]
        if type2_B != 'none' and type2_B in types[type2_A]:
            score *= types[type2_A][type2_B]

    return score

# this considers the possible choices for pokemon B and checks the matchup scores
# this loop is so expensive lol parallelization would be awesome here
expanded_data = []
count = 0
for index, row in df.iterrows():

    count += 1
    if count % 50 == 0:
        print(count)
    
    for b_choice in pokeB_choices:
        temp_row = row.copy()  # copy row for pokemon A
        temp_row['Pokemon_B'] = b_choice

        # Fetch the corresponding data for pokemon B
        pokemon_b_data = df[df['Pokemon Name'] == b_choice].iloc[0]
        
        # Ensure both pokemon A and B's types are lowercase and cleaned
        temp_row['Type1_A'] = temp_row['Primary Type'].strip().lower()
        temp_row['Type2_A'] = temp_row['Secondary Type'].strip().lower()
        temp_row['Type1_B'] = pokemon_b_data['Primary Type'].strip().lower()
        temp_row['Type2_B'] = pokemon_b_data['Secondary Type'].strip().lower()

        # If either Type2_A or Type2_B is missing or 'None', replace with 'none'
        temp_row['Type2_A'] = 'none' if temp_row['Type2_A'] == 'none' or pd.isnull(temp_row['Type2_A']) else temp_row['Type2_A']
        temp_row['Type2_B'] = 'none' if temp_row['Type2_B'] == 'none' or pd.isnull(temp_row['Type2_B']) else temp_row['Type2_B']

        # Calculate the type advantage score for pokemon A vs. pokemon B
        temp_row['Advantage_Score'] = calculate_type_advantage(
            temp_row['Type1_A'], temp_row['Type2_A'], temp_row['Type1_B'], temp_row['Type2_B'])
        
        expanded_data.append(temp_row)

# appending the new dataframe
expanded_df = pd.DataFrame(expanded_data)

# one hot encoding the type matchups
one_hot_types = pd.get_dummies(expanded_df[[
    'Type1_A', 'Type2_A', 'Type1_B', 'Type2_B']], 
    prefix=['Type1_A', 'Type2_A', 'Type1_B', 'Type2_B'])

# check numerical features
if 'Advantage_Score' in expanded_df.columns:
    X = pd.concat([one_hot_types, expanded_df[['Advantage_Score']]], axis=1)
else:
    X = one_hot_types


# encoding and formatting the data
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(expanded_df['Pokemon_B'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# change type to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# batching prep
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

train_dataset = train_dataset.to('device')
test_dataset = test_dataset.to('device')
train_loader = train_loader.to('device')
test_loader = test_loader.to('device')


class PokemonBattleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PokemonBattleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # first hidden layer
        self.fc2 = nn.Linear(128, 64)          # second hidden layer
        self.fc3 = nn.Linear(64, output_size)  # output layer
        self.relu = nn.ReLU()                  # activation function, might tweak this
        self.softmax = nn.Softmax(dim=1)       # softmax for probabilities
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))         
        return x

# initialize model
input_size = X_train.shape[1]
output_size = len(label_encoder.classes_)

model = PokemonBattleModel(input_size=input_size, output_size=output_size).to('device') # native gpu support or what

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training and evaluation
epochs = 25
for epoch in range(epochs):
    model.train()
    rolling_loss = 0.0
    pass

# do something here
# forward pass
# backprop
# test and score loop
