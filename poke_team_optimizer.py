
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pokémon Team Optimizer (Heuristic or RL)
----------------------------------------
Selects an optimal team of six Pokémon against a given opponent team, using:
  1) Pokémon stats CSV (cleaned automatically with requested column drops)
  2) Type matchup matrix CSV (Attacking x Defending multipliers)
  3) Opponent team of 6 Pokémon names
  4) Toggle to include/exclude Legendaries

Methods:
  - heuristic: neural scorer trained on synthetic labels + hill climbing search
  - rl       : REINFORCE policy that sequentially picks 6 Pokémon without replacement
"""
import argparse
import json
import math
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

TYPES = [
    "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison",
    "Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"
]
type_to_idx = {t:i for i,t in enumerate(TYPES)}
STAT_NAMES = ["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def one_hot_types(primary: str, secondary: str) -> np.ndarray:
    v = np.zeros(len(TYPES), dtype=np.float32)
    if isinstance(primary, str) and primary in type_to_idx: v[type_to_idx[primary]] = 1.0
    if isinstance(secondary, str) and secondary in type_to_idx and secondary != primary: v[type_to_idx[secondary]] = 1.0
    return v

def load_type_matrix(path: str) -> np.ndarray:
    df = pd.read_csv(path).set_index("Attacking"); df = df.loc[TYPES, TYPES]
    return df.to_numpy(dtype=np.float32)

def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns: return c
    return None

def drop_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        'Pokemon Id', 'Classification', 'Alternate Form Name', 'Original Pokemon ID',
        'Legendary Type', 'Pokemon Height', 'Pokemon Weight', 'Primary Ability Description',
        'Secondary Ability', 'Secondary Ability Description', 'Hidden Ability',
        'Hidden Ability Description', 'Special Event Ability', 'Special Event Ability Description',
        'Male Ratio', 'Female Ratio', 'Base Happiness', 'Game(s) of Origin', 'Catch Rate', 'Experience Growth',
        'Experience Growth Total', 'Primary Egg Group', 'Secondary Egg Group', 'Egg Cycle Count',
        'Pre-Evolution Pokemon Id', 'Evolution Details'
    ]
    drop_existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_existing)

def build_canon(df: pd.DataFrame) -> pd.DataFrame:
    name_col = find_col(df, ["Pokemon","Pokemon Name","Name"])
    t1_col = find_col(df, ["Primary Type","Type 1","Type1"])
    t2_col = find_col(df, ["Secondary Type","Type 2","Type2","Secondary type"])
    leg_col = find_col(df, ["Legendary","Is Legendary","Mythical"])
    stat_map = {
        "HP": ["HP","Base HP"],
        "Attack": ["Attack","Atk","Base Attack"],
        "Defense": ["Defense","Def","Base Defense"],
        "Sp. Atk": ["Special Attack","Sp. Atk","SpAtk","SpAttack","Base Sp. Atk","Sp Atk"],
        "Sp. Def": ["Special Defense","Sp. Def","SpDef","SpDefense","Base Sp. Def","Sp Def"],
        "Speed": ["Speed","Spe","Base Speed"]
    }
    def find_stat(cands):
        for c in cands:
            if c in df.columns: return c
        return None
    out = pd.DataFrame()
    out["Name"] = df[name_col].astype(str).str.strip('"') if name_col else df.index.astype(str)
    out["Type1"] = df[t1_col].astype(str).str.strip('"') if t1_col else ""
    out["Type2"] = df[t2_col].astype(str).str.strip('"') if t2_col else ""

    if leg_col:
        out["Legendary"] = df[leg_col].astype(str).str.lower().isin(["true","1","yes","legendary","mythical"]).astype(int)
    else:
        out["Legendary"] = 0
    for k, cands in stat_map.items():
        col = find_stat(cands); out[k] = pd.to_numeric(df[col], errors="coerce") if col else 0
    out = out.dropna(subset=["Name","Type1"]).reset_index(drop=True)
    for s in STAT_NAMES:
        mn, mx = out[s].min(), out[s].max()
        out[s] = (out[s]-mn)/(mx-mn) if mx>mn else 0.0
    return out

class BattleScorer:
    def __init__(self, canon: pd.DataFrame, eff_table: np.ndarray):
        self.canon = canon; self.eff_table = eff_table
        self.name_to_idx = {n.lower(): i for i,n in enumerate(canon["Name"].astype(str))}
    def _type_mult(self, atk_types: List[str], defend_types: List[str]) -> float:
        atk_idx = [type_to_idx.get(t) for t in atk_types if t in type_to_idx]
        def_idx = [type_to_idx.get(t) for t in defend_types if t in type_to_idx]
        if not atk_idx or not def_idx: return 1.0
        best = 0.0
        for ai in atk_idx:
            mult = 1.0
            for di in def_idx: mult *= float(self.eff_table[ai, di])
            best = max(best, mult)
        return best if best>0 else 0.0
    def versus_score(self, attacker_idx: int, defender_idx: int) -> float:
        t1 = str(self.canon.loc[attacker_idx,"Type1"]); t2 = str(self.canon.loc[attacker_idx,"Type2"])
        dt1 = str(self.canon.loc[defender_idx,"Type1"]); dt2 = str(self.canon.loc[defender_idx,"Type2"])
        atk_types = [t1] + ([t2] if t2 and t2 != "nan" and t2 != "" else [])
        def_types = [dt1] + ([dt2] if dt2 and dt2 != "nan" and dt2 != "" else [])
        eff = self._type_mult(atk_types, def_types)
        off = float(self.canon.loc[attacker_idx, ["Attack","Sp. Atk","Speed"]].mean())
        dfn = float(self.canon.loc[attacker_idx, ["HP","Defense","Sp. Def"]].mean())
        return 0.7*eff + 0.2*off + 0.1*dfn
    def team_score(self, team_idxs: List[int], opp_idxs: List[int]) -> float:
        remaining = opp_idxs.copy(); total = 0.0
        for i in team_idxs:
            best = 0.0; best_j = None
            for j in remaining:
                s = self.versus_score(i, j)
                if s > best: best, best_j = s, j
            total += best
            if best_j is not None: remaining.remove(best_j)
            if not remaining: break
        denom = max(1, len(opp_idxs))
        return total/denom

def build_features(canon: pd.DataFrame) -> np.ndarray:
    feats = []
    for _, row in canon.iterrows():
        tvec = one_hot_types(str(row["Type1"]), str(row["Type2"]))
        stats = row[STAT_NAMES].values.astype(np.float32)
        feats.append(np.concatenate([tvec, stats], axis=0))
    return np.stack(feats, axis=0)

class ScorerNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def opponent_vector(canon: pd.DataFrame, opp_idxs: List[int]) -> np.ndarray:
    t_agg = np.zeros(len(TYPES), dtype=np.float32)
    s_agg = np.zeros(len(STAT_NAMES), dtype=np.float32)
    if not opp_idxs: return np.concatenate([t_agg, s_agg], axis=0)
    for j in opp_idxs:
        t_agg += one_hot_types(str(canon.loc[j,"Type1"]), str(canon.loc[j,"Type2"]))
        s_agg += canon.loc[j, STAT_NAMES].values.astype(np.float32)
    t_agg /= len(opp_idxs); s_agg /= len(opp_idxs)
    return np.concatenate([t_agg, s_agg], axis=0)

def synthetic_pairs(X: np.ndarray, canon: pd.DataFrame, opp_pool: List[int], scorer: BattleScorer,
                    num_pairs: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    n = len(canon)
    inputs, labels = [], []
    for _ in range(num_pairs):
        team_size = min(6, len(opp_pool))
        opp_idxs = list(rng.choice(opp_pool, size=team_size, replace=False))
        cand = int(rng.integers(0, n))
        opp_vec = opponent_vector(canon, opp_idxs)
        pair = np.concatenate([X[cand], opp_vec], axis=0)
        y = np.mean([scorer.versus_score(cand, j) for j in opp_idxs]) if opp_idxs else 0.0
        inputs.append(pair); labels.append(y)
    return np.stack(inputs), np.array(labels, dtype=np.float32)

def heuristic_optimize(canon: pd.DataFrame, eff_table: np.ndarray, opp_names: List[str],
                       include_legendaries: bool = True, steps: int = 150, diversity_penalty: float = 0.12,
                       synth_pairs: int = 3000, seed: int = 42) -> List[Tuple[str,float]]:
    set_seed(seed)
    scorer = BattleScorer(canon, eff_table)
    name_to_idx = {n.lower(): i for i,n in enumerate(canon["Name"].astype(str))}
    opp_idxs = [name_to_idx[n.lower()] for n in opp_names if n.lower() in name_to_idx]
    pool = [i for i, r in canon.iterrows() if include_legendaries or int(r["Legendary"])==0]
    X = build_features(canon)
    train_X, train_y = synthetic_pairs(X, canon, pool, scorer, num_pairs=synth_pairs)
    val_X, val_y = synthetic_pairs(X, canon, pool, scorer, num_pairs=max(500, synth_pairs//5))

    in_dim = X.shape[1] * 2
    model = ScorerNN(in_dim); opt = optim.Adam(model.parameters(), lr=1e-3); mse = nn.MSELoss()
    tx = torch.tensor(train_X, dtype=torch.float32); ty = torch.tensor(train_y, dtype=torch.float32)
    vx = torch.tensor(val_X, dtype=torch.float32); vy = torch.tensor(val_y, dtype=torch.float32)
    for _ in range(6):
        model.train(); opt.zero_grad(); loss = mse(model(tx), ty); loss.backward(); opt.step()
    model.eval()

    opp_vec = opponent_vector(canon, opp_idxs)
    scores = []
    with torch.no_grad():
        for i in pool:
            x = torch.tensor(np.concatenate([X[i], opp_vec], axis=0), dtype=torch.float32)
            s = float(model(x.unsqueeze(0)).item())
            scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)

    picked = []; type_counts = np.zeros(len(TYPES), dtype=np.int32)
    def penalty(i):
        t1 = str(canon.loc[i,"Type1"]); t2 = str(canon.loc[i,"Type2"]); p = 0.0
        if t1 in type_to_idx: p += diversity_penalty * type_counts[type_to_idx[t1]]
        if t2 and t2 in type_to_idx and t2 != t1: p += diversity_penalty * type_counts[type_to_idx[t2]]
        return p
    for i, s in scores:
        if len(picked) >= 6: break
        picked.append(i)
        t1 = str(canon.loc[i,"Type1"]); t2 = str(canon.loc[i,"Type2"])
        if t1 in type_to_idx: type_counts[type_to_idx[t1]] += 1
        if t2 and t2 in type_to_idx and t2 != t1: type_counts[type_to_idx[t2]] += 1

    team = picked[:6]
    best_score = scorer.team_score(team, opp_idxs)

    for _ in range(steps):
        out_idx = random.choice(team); cand_idx = random.choice(pool)
        if cand_idx in team: continue
        new_team = [cand_idx if x==out_idx else x for x in team]
        new_score = scorer.team_score(new_team, opp_idxs)
        if new_score > best_score:
            team = new_team; best_score = new_score

    per_member = []
    for i in team:
        s = np.mean([scorer.versus_score(i, j) for j in opp_idxs]) if opp_idxs else 0.0
        per_member.append((str(canon.loc[i,"Name"]), s))
    per_member.sort(key=lambda x: x[1], reverse=True)
    return per_member

class TeamPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()); self.head = nn.Linear(hidden,1)
    def forward(self, x): return self.head(self.enc(x)).squeeze(-1)

def rl_optimize(canon: pd.DataFrame, eff_table: np.ndarray, opp_names: List[str],
                include_legendaries: bool = True, episodes: int = 400, seed: int = 42) -> List[Tuple[str,float]]:
    set_seed(seed)
    scorer = BattleScorer(canon, eff_table)
    name_to_idx = {n.lower(): i for i,n in enumerate(canon["Name"].astype(str))}
    opp_idxs = [name_to_idx[n.lower()] for n in opp_names if n.lower() in name_to_idx]
    pool = [i for i, r in canon.iterrows() if include_legendaries or int(r["Legendary"])==0]
    if len(pool) < 6: raise ValueError("Not enough Pokémon in pool after legendary filter.")
    X = build_features(canon); opp_vec = opponent_vector(canon, opp_idxs)
    in_dim = X.shape[1] + opp_vec.shape[0]
    policy = TeamPolicy(in_dim); optimizer = optim.Adam(policy.parameters(), lr=2e-3); baseline = None

    for _ in range(episodes):
        remaining = pool.copy(); chosen = []; logps = []
        cand_feats = [np.concatenate([X[i], opp_vec], axis=0) for i in remaining]
        for _step in range(6):
            feats = torch.tensor(np.stack(cand_feats), dtype=torch.float32)
            logits = policy(feats); probs = torch.softmax(logits, dim=0)
            m = torch.distributions.Categorical(probs=probs); idx = m.sample()
            logps.append(m.log_prob(idx)); pick = int(idx.item())
            chosen.append(remaining[pick]); remaining.pop(pick); cand_feats.pop(pick)
        R = scorer.team_score(chosen, opp_idxs)
        baseline = R if baseline is None else 0.9*baseline + 0.1*R
        adv = R - baseline
        loss = -adv * torch.stack(logps).sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    remaining = pool.copy(); chosen = []; cand_feats = [np.concatenate([X[i], opp_vec], axis=0) for i in remaining]
    with torch.no_grad():
        for _ in range(6):
            feats = torch.tensor(np.stack(cand_feats), dtype=torch.float32)
            logits = policy(feats); pick = int(torch.argmax(logits).item())
            chosen.append(remaining[pick]); remaining.pop(pick); cand_feats.pop(pick)

    per_member = []
    for i in chosen:
        s = np.mean([scorer.versus_score(i, j) for j in opp_idxs]) if opp_idxs else 0.0
        per_member.append((str(canon.loc[i,"Name"]), s))
    per_member.sort(key=lambda x: x[1], reverse=True)
    return per_member

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokemon_csv", type=str, required=True)
    ap.add_argument("--types_csv", type=str, required=True)
    ap.add_argument("--opp", type=str, nargs="+", required=True)
    ap.add_argument("--method", type=str, choices=["heuristic","rl"], default="heuristic")
    ap.add_argument("--no_legs", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heuristic_steps", type=int, default=150)
    ap.add_argument("--heuristic_pairs", type=int, default=3000)
    ap.add_argument("--diversity_penalty", type=float, default=0.12)
    ap.add_argument("--rl_episodes", type=int, default=400)
    args = ap.parse_args()

    set_seed(args.seed)
    raw = pd.read_csv(args.pokemon_csv); raw = drop_unused_cols(raw); canon = build_canon(raw)
    eff_table = load_type_matrix(args.types_csv)

    include_legendaries = not args.no_legs
    if args.method == "heuristic":
        team = heuristic_optimize(
            canon, eff_table, args.opp, include_legendaries=include_legendaries,
            steps=args.heuristic_steps, diversity_penalty=args.diversity_penalty,
            synth_pairs=args.heuristic_pairs, seed=args.seed
        )
    else:
        team = rl_optimize(
            canon, eff_table, args.opp, include_legendaries=include_legendaries,
            episodes=args.rl_episodes, seed=args.seed
        )

    print(json.dumps({
        "opponent": args.opp,
        "include_legendaries": include_legendaries,
        "method": args.method,
        "team": [{"name": n, "member_score": round(float(s), 4)} for n, s in team]
    }, indent=2))

import pandas as pd

# paths
POKEMON_CSV = "pokemon.csv"
TYPES_CSV = "types.csv"

def optimize_team(opponent_names, include_legendaries=True, method="heuristic"):
    """
    GUI-friendly wrapper to run optimizer and return team as list of dicts.
    """
    raw = pd.read_csv(POKEMON_CSV)
    raw = drop_unused_cols(raw)
    canon = build_canon(raw)
    eff_table = load_type_matrix(TYPES_CSV)

    if method == "heuristic":
        team = heuristic_optimize(canon, eff_table, opponent_names, include_legendaries=include_legendaries)
    else:
        team = rl_optimize(canon, eff_table, opponent_names, include_legendaries=include_legendaries)

    # Return list of dicts with 'name' and 'score'
    return [{"name": n, "score": float(s)} for n, s in team]

if __name__ == "__main__":
    main()