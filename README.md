# pokemonOptimizer
Given an opponents Pokemon team, this will return the Pokemon most likely to beat the opponenet team in battle. Used heuristic optimization methods
and alternatively, an RL approach. The script here is just using the heuristic optimization however.
I won't go through how to download the sprites but a simple script and the sprites can be downloaded from [here](https://pokemondb.net/sprites). Put them in a directory in the root named ``sprites``. From there you will just need to build and run it by running the following in bash/Git bash:
```pyinstaller \
  --name "PokemonOptimizer" \
  --noconsole \
  --collect-all torch \
  --add-data "pokemon.csv;." \
  --add-data "loading.gif;." \
  --add-data "sprites;sprites" \
  gui.py
```

Then run ``./dist/PokemonOptimizer/PokemonOptimizer.exe``. This is still in a very early state 
