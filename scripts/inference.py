#!/usr/bin/env python3
"""
Inductive GNN inference script.

Recommend top-K tires for an existing user or a **new** user. For new users,
the script injects a temporary node into the heterogeneous graph with
preference-based edges, then runs the full HGT → Intermediate → FusionMLP
pipeline (Inductive GNN) — the same model used during training.

Usage
-----
    # Existing user (by index):
    uv run python scripts/inference.py --user 42

    # New user with structured preferences:
    uv run python scripts/inference.py --user new \\
        --brand "Michelin,Continental" \\
        --size "235/40R18" \\
        --budget-min 100 --budget-max 200 \\
        --min-treadwear 400 \\
        --traction A \\
        --temperature A
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TireRecommender


# ─── Utilities ────────────────────────────────────────────────────────

def pick_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# UTQG grade ordering (higher is better).
_TRACTION_ORDER = {"AA": 3, "A": 2, "B": 1, "C": 0, "Unknown": -1}
_TEMPERATURE_ORDER = {"A": 2, "B": 1, "C": 0, "Unknown": -1}


def _grade_ge(grade: str, threshold: str, ordering: dict[str, int]) -> bool:
    """Return True if ``grade`` is >= ``threshold`` in the given ordering."""
    return ordering.get(grade, -1) >= ordering.get(threshold, -1)


def find_matching_tires(
    tire_df: pd.DataFrame,
    tire_map: dict[str, int],
    preferences: dict,
) -> list[int]:
    """
    Filter the tire catalog by the user's structured preferences.

    Returns a list of graph-level tire node indices that satisfy ALL
    specified filters (AND logic).
    """
    mask = pd.Series(True, index=tire_df.index)

    if preferences.get("brands"):
        brands_upper = [b.upper() for b in preferences["brands"]]
        mask &= tire_df["brand"].str.upper().isin(brands_upper)

    if preferences.get("size"):
        mask &= tire_df["size"] == preferences["size"]

    if preferences.get("budget_min") is not None:
        mask &= tire_df["price"].fillna(0) >= preferences["budget_min"]

    if preferences.get("budget_max") is not None:
        mask &= tire_df["price"].fillna(float("inf")) <= preferences["budget_max"]

    if preferences.get("min_treadwear") is not None:
        mask &= tire_df["treadwear"].fillna(0) >= preferences["min_treadwear"]

    if preferences.get("traction"):
        mask &= tire_df["traction"].apply(
            lambda g: _grade_ge(g, preferences["traction"], _TRACTION_ORDER)
        )

    if preferences.get("temperature"):
        mask &= tire_df["temperature"].apply(
            lambda g: _grade_ge(g, preferences["temperature"], _TEMPERATURE_ORDER)
        )

    # Map surviving tire_titles to graph node indices.
    matched_titles = tire_df.loc[mask, "tire_title"]
    return [tire_map[t] for t in matched_titles if t in tire_map]


# ─── Inductive GNN: inject new user into graph ───────────────────────

@torch.no_grad()
def recommend_existing_user(
    model: TireRecommender,
    data,
    user_idx: int,
    k: int,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    """Score all tires for an existing user and return top-K."""
    model.eval()
    out = model.encode(data)
    num_tires = out["h_tire_t"].size(0)
    device = out["h_tire_t"].device
    all_tires = torch.arange(num_tires, device=device)

    users_rep = torch.full((num_tires,), user_idx, dtype=torch.long, device=device)
    scores = model.score(out, users_rep, all_tires)

    topk_vals, topk_idx = torch.topk(scores, k=k)
    cluster_dists = out["C_tire"].cpu()

    results = []
    for rank, (t_idx, score) in enumerate(
        zip(topk_idx.cpu().tolist(), topk_vals.cpu().tolist()), start=1
    ):
        top_cluster = max(range(cluster_dists.size(1)),
                          key=lambda i: cluster_dists[t_idx][i].item())
        results.append({
            "rank": rank, "tire_idx": t_idx,
            "score": round(score, 4), "top_cluster": top_cluster,
        })
    return results, out


@torch.no_grad()
def recommend_new_user(
    model: TireRecommender,
    data,
    matching_tires: list[int],
    k: int,
    default_rating: float = 4.5,
) -> list[dict]:
    """
    Inductive GNN inference for a new user.

    1. Deep-copy the graph.
    2. Add a new user node (initialized with the mean of all existing user
       embeddings) and preference edges to matching tires.
    3. Run the full HGT → Intermediate → FusionMLP pipeline.
    4. Return top-K scored tires for the new user.
    """
    # ── 1. Deep-copy graph so we don't mutate the original. ──────────
    data = copy.deepcopy(data)
    device = data["tire"].x.device

    new_user_idx = data["user"].num_nodes  # e.g., 12869
    data["user"].num_nodes = new_user_idx + 1
    data["user"].node_id = torch.arange(new_user_idx + 1, device=device)

    # ── 2. Expand user embedding table with mean-init row. ───────────
    old_emb = model.encoder.input_emb["user"]                 # nn.Embedding
    old_weights = old_emb.weight.data                          # (N, 128)
    mean_row = old_weights.mean(dim=0, keepdim=True)           # (1, 128)
    new_weights = torch.cat([old_weights, mean_row], dim=0)    # (N+1, 128)

    # Temporarily replace the embedding table.
    orig_emb = model.encoder.input_emb["user"]
    model.encoder.input_emb["user"] = nn.Embedding.from_pretrained(
        new_weights, freeze=True
    ).to(device)

    # ── 3. Add preference edges (user ↔ matching tires). ─────────────
    matching = torch.tensor(matching_tires, dtype=torch.long, device=device)
    n_match = matching.size(0)
    new_user_col = torch.full((n_match,), new_user_idx, dtype=torch.long, device=device)

    # Forward: (user, reviews, tire)
    new_fw = torch.stack([new_user_col, matching])
    old_fw = data["user", "reviews", "tire"].edge_index.to(device)
    data["user", "reviews", "tire"].edge_index = torch.cat([old_fw, new_fw], dim=1)

    # edge_attr for new edges (default high rating)
    old_attr = data["user", "reviews", "tire"].edge_attr.to(device)
    new_attr = torch.full((n_match, 1), default_rating, device=device)
    data["user", "reviews", "tire"].edge_attr = torch.cat([old_attr, new_attr], dim=0)

    # Reverse: (tire, rev_by, user)
    new_rv = torch.stack([matching, new_user_col])
    old_rv = data["tire", "rev_by", "user"].edge_index.to(device)
    data["tire", "rev_by", "user"].edge_index = torch.cat([old_rv, new_rv], dim=1)

    old_rv_attr = data["tire", "rev_by", "user"].edge_attr.to(device)
    data["tire", "rev_by", "user"].edge_attr = torch.cat([old_rv_attr, new_attr], dim=0)

    # ── 4. Full HGT forward → Intermediate → FusionMLP. ─────────────
    model.eval()
    out = model.encode(data)

    num_tires = out["h_tire_t"].size(0)
    all_tires = torch.arange(num_tires, device=device)
    users_rep = torch.full((num_tires,), new_user_idx, dtype=torch.long, device=device)
    scores = model.score(out, users_rep, all_tires)

    topk_vals, topk_idx = torch.topk(scores, k=k)
    cluster_dists = out["C_tire"].cpu()

    results = []
    for rank, (t_idx, score) in enumerate(
        zip(topk_idx.cpu().tolist(), topk_vals.cpu().tolist()), start=1
    ):
        top_cluster = max(range(cluster_dists.size(1)),
                          key=lambda i: cluster_dists[t_idx][i].item())
        results.append({
            "rank": rank, "tire_idx": t_idx,
            "score": round(score, 4), "top_cluster": top_cluster,
        })

    # ── 5. Restore original embedding table. ─────────────────────────
    model.encoder.input_emb["user"] = orig_emb

    return results


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recommend tires using Inductive GNN inference."
    )
    p.add_argument(
        "--user", type=str, required=True,
        help='User index (int) for an existing user, or "new" for a cold-start user.',
    )
    p.add_argument("--checkpoint", type=str,
                    default="outputs/checkpoints/recommender_e8.pt")
    p.add_argument("--k", type=int, default=10, help="Number of recommendations.")

    # Model architecture (must match training).
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-clusters", type=int, default=50)

    # New-user preference filters.
    p.add_argument("--brand", type=str, default=None,
                    help='Comma-separated preferred brands, e.g. "Michelin,Continental".')
    p.add_argument("--size", type=str, default=None,
                    help='Tire size, e.g. "235/40R18".')
    p.add_argument("--budget-min", type=float, default=None)
    p.add_argument("--budget-max", type=float, default=None)
    p.add_argument("--min-treadwear", type=float, default=None,
                    help="Minimum UTQG treadwear rating.")
    p.add_argument("--traction", type=str, default=None,
                    help='Minimum traction grade: AA, A, B, or C.')
    p.add_argument("--temperature", type=str, default=None,
                    help='Minimum temperature grade: A, B, or C.')

    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load graph + mappings + tire metadata ────────────────────────
    graph_path = PROJECT_ROOT / "data" / "processed" / "hetero_graph.pt"
    print(f"Loading graph from {graph_path} ...")
    payload = torch.load(graph_path, weights_only=False)
    data = payload["graph"]
    mappings = payload["mappings"]
    tire_df: pd.DataFrame = payload["tire_df"]

    device = pick_device(args.device)
    data = data.to(device)

    # Reverse maps: node index → human-readable name.
    idx_to_tire = {v: k for k, v in mappings["tire_map"].items()}
    idx_to_brand = {v: k for k, v in mappings["brand_map"].items()}

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
    model.eval()

    num_users = data["user"].num_nodes
    num_tires = data["tire"].num_nodes

    # ── Route: existing user vs. new user ────────────────────────────
    if args.user.lower() == "new":
        # Build preferences dict from CLI flags.
        preferences: dict = {}
        if args.brand:
            preferences["brands"] = [b.strip() for b in args.brand.split(",")]
        if args.size:
            preferences["size"] = args.size
        if args.budget_min is not None:
            preferences["budget_min"] = args.budget_min
        if args.budget_max is not None:
            preferences["budget_max"] = args.budget_max
        if args.min_treadwear is not None:
            preferences["min_treadwear"] = args.min_treadwear
        if args.traction:
            preferences["traction"] = args.traction.upper()
        if args.temperature:
            preferences["temperature"] = args.temperature.upper()

        if not preferences:
            print("Error: --user new requires at least one preference flag "
                  "(--brand, --size, --budget-min, etc.)")
            sys.exit(1)

        print(f"\nNew user preferences:")
        for key, val in preferences.items():
            print(f"  {key}: {val}")

        matching = find_matching_tires(tire_df, mappings["tire_map"], preferences)
        print(f"\nMatching tires in catalog: {len(matching)} / {num_tires}")

        if not matching:
            print("Error: no tires match the given preferences. Try relaxing filters.")
            sys.exit(1)

        print(f"Injecting new user node with {len(matching)} preference edges ...")
        results = recommend_new_user(model, data, matching, k=args.k)

    else:
        user_idx = int(args.user)
        if user_idx < 0 or user_idx >= num_users:
            print(f"Error: user index must be in [0, {num_users - 1}]")
            sys.exit(1)
        print(f"\nExisting user: {user_idx}")
        results, _ = recommend_existing_user(model, data, user_idx, k=args.k)

    # ── Print results ────────────────────────────────────────────────
    print(f"\nTop-{args.k} Recommended Tires:")
    print(f"{'Rank':<6} {'Score':<10} {'Cluster':<10} {'Tire'}")
    print("-" * 80)
    for rec in results:
        tire_name = idx_to_tire.get(rec["tire_idx"], f"tire_{rec['tire_idx']}")
        # Truncate long tire names for display.
        if len(tire_name) > 55:
            tire_name = tire_name[:52] + "..."
        print(
            f"{rec['rank']:<6} {rec['score']:<10.4f} "
            f"{'cluster ' + str(rec['top_cluster']):<10} {tire_name}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
