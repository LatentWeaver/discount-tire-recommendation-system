#!/usr/bin/env python3
"""
Evaluate the trained TireRecommender model.

Computes Recall@K, NDCG@K, and HitRate@K for:
  1. Standard Mode: users evaluated with embeddings produced from the
     train-only review graph.
  2. Cold-Start Mode: simulates test users as brand new
     users with preference filters only (brand, size, budget) extracted
     from their train-split history.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference import find_matching_tires, recommend_new_user
from src.models import TireRecommender
from src.training.evaluation import evaluate as evaluate_ranking
from src.training.sampler import BPRSampler


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
    user_train_positive_lists: list[list[int]],
    tire_df: pd.DataFrame,
    mappings: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Extract a coarse preference summary from a user's train split."""
    train_tires = user_train_positive_lists[user_idx]
    if not train_tires:
        return {}

    idx_to_tire = {v: k for k, v in mappings["tire_map"].items()}
    train_titles = [idx_to_tire.get(t_idx) for t_idx in train_tires]

    user_bought = tire_df[tire_df["tire_title"].isin(train_titles)]
    if user_bought.empty:
        return {}

    brands = user_bought["brand"].unique().tolist()
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


def init_metric_sums(ks: tuple[int, ...]) -> dict[str, float]:
    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"HitRate@{k}": 0.0 for k in ks})
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    return sums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/recommender_e8.pt",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--ks",
        type=str,
        default="10,20,50",
        help="Comma-separated ranking cutoffs, e.g. 10,20,50",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Single cutoff shortcut; overrides --ks when provided.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-clusters", type=int, default=50)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed used to reconstruct the train-only review graph.",
    )
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
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_clusters=args.num_clusters,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    print(f"Loading checkpoint from {ckpt_path} ...")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # We need the sampler for its train/test splits
    print(
        f"Initializing sampler to retrieve train/test splits "
        f"(seed={args.split_seed})..."
    )
    torch.manual_seed(args.split_seed)
    sampler = BPRSampler(
        data,
        rating_threshold=args.rating_threshold,
        seed=args.split_seed,
    )

    eval_data = sampler.train_data
    user_train_positive_lists = sampler.user_positives_list
    if args.k is not None:
        ks = (args.k,)
    else:
        ks = tuple(sorted({int(k.strip()) for k in args.ks.split(",") if k.strip()}))
        if not ks:
            raise ValueError("At least one evaluation cutoff must be provided.")

    print("\n" + "=" * 50)
    print(" 1. Standard Testing Mode (Existing Users)")
    print("=" * 50)

    test_metrics = evaluate_ranking(
        model,
        eval_data,
        sampler.test_users,
        sampler.test_tires,
        user_train_positive_lists,
        ks=ks,
    )

    for k, v in test_metrics.items():
        print(f"  {k:<14s} {v:.4f}")

    print("\n" + "=" * 50)
    print(" 2. Cold-Start Simulation")
    print("=" * 50)
    print("Simulating test users as new users using extracted training preferences...")

    test_users = sampler.test_users
    test_tires = sampler.test_tires
    n_simulate = min(500, test_users.size(0))
    sums = init_metric_sums(ks)

    valid_simulations = 0
    import tqdm

    for i in tqdm.tqdm(range(n_simulate), desc="Simulating cold starts"):
        u = int(test_users[i])
        gt = int(test_tires[i])

        pref = extract_preferences_for_simulation(
            u,
            user_train_positive_lists,
            tire_df,
            mappings,
        )
        if not pref:
            continue

        matching = find_matching_tires(tire_df, mappings["tire_map"], pref)
        if not matching:
            continue

        recs = recommend_new_user(model, eval_data, matching, k=max(ks))
        topk = [r["tire_idx"] for r in recs]

        valid_simulations += 1

        for k in ks:
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
