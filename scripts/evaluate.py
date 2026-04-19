#!/usr/bin/env python3
"""
Evaluate the trained TireRecommender model.

Computes Recall@K, NDCG@K, and HitRate@K for:
  1. Standard Mode: users evaluated using their full graph embedding
     learned from training history.
  2. Inductive Cold-Start Mode: simulates test users as brand new
     users with preference filters only (brand, size, budget) extracted
     from their training history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TireRecommender
from src.training.trainer import Trainer
from scripts.inference import recommend_new_user


def pick_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_preferences_for_simulation(
    user_idx: int,
    user_positives_train: list[set[int]],
    tire_df: pd.DataFrame,
    mappings: dict,
) -> dict:
    """Extract preference summary from a user's training history."""
    train_tires = list(user_positives_train[user_idx])
    if not train_tires:
        return {}

    idx_to_tire = {v: k for k, v in mappings["tire_map"].items()}
    train_titles = [idx_to_tire.get(t_idx) for t_idx in train_tires]
    
    # Filter tire_df to only what the user bought in train split
    user_bought = tire_df[tire_df["tire_title"].isin(train_titles)]
    if user_bought.empty:
        return {}

    # Extract summarize preferences
    brands = user_bought["brand"].unique().tolist()
    
    # get most common size
    sizes = user_bought["size"].value_counts()
    size = sizes.index[0] if not sizes.empty else None

    budget_min = user_bought["price"].min()
    budget_max = user_bought["price"].max()
    
    return {
        "brands": brands,
        "size": size,
        "budget_min": budget_min if pd.notnull(budget_min) else None,
        "budget_max": budget_max if pd.notnull(budget_max) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/recommender_e8.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    graph_path = PROJECT_ROOT / "data" / "processed" / "hetero_graph.pt"
    print(f"Loading graph from {graph_path} ...")
    payload = torch.load(graph_path, weights_only=False)
    data = payload["graph"]
    mappings = payload["mappings"]
    tire_df = payload["tire_df"]

    device = pick_device(args.device)
    data = data.to(device)

    # ── Load model ───────────────────────────────────────────────────
    model = TireRecommender.from_data(
        data,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_clusters=50,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    print(f"Loading checkpoint from {ckpt_path} ...")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    from src.training.sampler import BPRSampler
    from src.training.evaluation import evaluate as std_evaluate

    # We need the sampler for its train/test splits
    print("Initializing sampler to retrieve train/test splits (seed=0)...")
    torch.manual_seed(0)
    sampler = BPRSampler(data)

    print("\n" + "="*50)
    print(" 1. Standard Testing Mode (Existing Users)")
    print("="*50)
    
    test_metrics = std_evaluate(
        model, data, sampler.test_users, sampler.test_tires, sampler.user_positives_list, ks=(10, 20, 50)
    )

    for k, v in test_metrics.items():
        print(f"  {k:<14s} {v:.4f}")

    print("\n" + "="*50)
    print(" 2. Inductive Cold-Start Simulation")
    print("="*50)
    print("Simulating test users as new users using extracted training preferences...")
    
    # Take a subset of test users to simulate so it doesn't take forever
    test_users = sampler.test_users
    test_tires = sampler.test_tires
    n_simulate = min(500, test_users.size(0))
    import math
    
    sums = {f"Recall@{k}": 0.0 for k in [10, 20, 50]}
    sums.update({f"HitRate@{k}": 0.0 for k in [10, 20, 50]})
    sums.update({f"NDCG@{k}": 0.0 for k in [10, 20, 50]})
    
    from scripts.inference import find_matching_tires

    valid_simulations = 0
    import tqdm
    for i in tqdm.tqdm(range(n_simulate), desc="Simulating cold starts"):
        u = int(test_users[i])
        gt = int(test_tires[i])
        
        pref = extract_preferences_for_simulation(
            u, sampler.user_positives_list, tire_df, mappings
        )
        if not pref:
            continue
            
        matching = find_matching_tires(tire_df, mappings["tire_map"], pref)
        if not matching:
            continue
            
        recs = recommend_new_user(model, data, matching, k=max(10, 20, 50))
        topk = [r["tire_idx"] for r in recs]
        
        valid_simulations += 1
        
        for k in [10, 20, 50]:
            in_top = gt in topk[:k]
            sums[f"Recall@{k}"] += float(in_top)
            sums[f"HitRate@{k}"] += float(in_top)
            if in_top:
                rank = topk[:k].index(gt)
                sums[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)

    print(f"\nSimulated metrics over {valid_simulations} test cases:")
    for k, v in sums.items():
        print(f"  {k:<14s} {v / valid_simulations:.4f}")

if __name__ == "__main__":
    main()
