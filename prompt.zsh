# Sample prompt for zsh shell to run the script and format output with jq
python poke_team_optimizer.py \
  --pokemon_csv pokemon.csv \
  --types_csv types.csv \
  --opp Pikachu Charizard Blastoise Venusaur Gengar Snorlax \
  --method heuristic \
  --no_legs | jq -r '.team[] | "\(.name) : \(.member_score)"'