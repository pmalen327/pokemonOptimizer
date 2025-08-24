import pandas as pd
import os
import requests
from bs4 import BeautifulSoup

# Load CSV
csv_path = "pokemon.csv"  # update if needed
df = pd.read_csv(csv_path)

name_col = "Pokemon Name"  # column with names

# Folder for sprites
os.makedirs("sprites", exist_ok=True)

# Special case mapping for PokémonDB slugs
special_cases = {
    "nidoran♀": "nidoran-f",
    "nidoran♂": "nidoran-m",
    "farfetch’d": "farfetchd",
    "mr. mime": "mr-mime",
    "mime jr.": "mime-jr",
    "type: null": "type-null",
    "jangmo-o": "jangmo-o",
    "hakamo-o": "hakamo-o",
    "kommo-o": "kommo-o",
    "tapu koko": "tapu-koko",
    "tapu lele": "tapu-lele",
    "tapu bulu": "tapu-bulu",
    "tapu fini": "tapu-fini",
    "flabébé": "flabebe",
    "sirfetch’d": "sirfetchd",
    "mr. rime": "mr-rime",
    "porygon-z": "porygon-z",
    "ho-oh": "ho-oh",
    "jangmo’o": "jangmo-o",
    "hakamo’o": "hakamo-o",
    "kommo’o": "kommo-o",
    "great tusk": "great-tusk",
    "scream tail": "scream-tail",
    "brute bonnet": "brute-bonnet",
    "flutter mane": "flutter-mane",
    "slither wing": "slither-wing",
    "sandy shocks": "sandy-shocks",
    "iron treads": "iron-treads",
    "iron bundle": "iron-bundle",
    "iron hands": "iron-hands",
    "iron jugulis": "iron-jugulis",
    "iron moth": "iron-moth",
    "iron thorns": "iron-thorns",
    "iron valiant": "iron-valiant",
    "wo-chien": "wo-chien",
    "chien-pao": "chien-pao",
    "ting-lu": "ting-lu",
    "chi-yu": "chi-yu",
    "roaring moon": "roaring-moon",
    "walking wake": "walking-wake",
    "iron leaves": "iron-leaves",
    "gouging fire": "gouging-fire",
    "raging bolt": "raging-bolt",
    "iron boulder": "iron-boulder",
    "iron crown": "iron-crown",
    "terapagos": "terapagos"
}

def clean_name(name: str) -> str:
    n = str(name).strip().lower().replace('"', '')
    n = n.replace(" ", "-")
    return special_cases.get(n, n)

pokemon_names = [clean_name(x) for x in df[name_col]]

# Download sprites
for name in pokemon_names:
    url = f"https://img.pokemondb.net/sprites/home/normal/{name}.png"
    response = requests.get(url)

    if response.status_code == 200:
        file_path = os.path.join("sprites", f"{name}.png")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded: {name}")
    else:
        print(f"❌ Failed: {name} ({url})")
