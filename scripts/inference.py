#!/usr/bin/env python3
"""
Two-Tower + FAISS inference.

Two modes (mirror §8 of new_instructions.md):
  1. Existing user (warm) — FAISS top-K against cached user_vec.
  2. Cold-start user      — inject a temporary user node into the train
                            graph with `reviews` edges to preference-matched
                            tires, run HGT → user tower, FAISS top-K.

Usage
-----
    # Existing user (by index):
    uv run python scripts/inference.py --user 42 --k 10

    # New user with structured preferences:
    uv run python scripts/inference.py --user new \\
        --brand "Michelin,Continental" \\
        --size "235/40R18" \\
        --budget-min 100 --budget-max 250 \\
        --min-treadwear 400 --traction A --temperature A
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# torch + faiss both ship with libomp on macOS; allow the duplicate.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TwoTowerRecommender
from src.training.sampler import BPRSampler
from src.training.trainer import build_tire_lookup


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
    return ordering.get(grade, -1) >= ordering.get(threshold, -1)


def find_matching_tires(
    tire_df: pd.DataFrame,
    tire_map: dict[str, int],
    preferences: dict,
) -> list[int]:
    """AND-filter the tire catalog by the user's structured preferences."""
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

    matched_titles = tire_df.loc[mask, "tire_title"]
    return [tire_map[t] for t in matched_titles if t in tire_map]


def faiss_search(
    index: faiss.Index,
    query: np.ndarray,
    k: int,
    masked: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Return list of (tire_idx, score), masked tires excluded post-hoc."""
    over_k = k + (len(masked) if masked else 0)
    over_k = min(over_k, index.ntotal)
    scores, idx = index.search(query.reshape(1, -1).astype("float32"), over_k)
    out: list[tuple[int, float]] = []
    for i, s in zip(idx[0].tolist(), scores[0].tolist()):
        if masked and i in masked:
            continue
        out.append((i, s))
        if len(out) >= k:
            break
    return out


@torch.no_grad()
def cold_start_user_vec(
    model: TwoTowerRecommender,
    train_data,
    matching_tires: list[int],
    default_rating: float = 4.5,
) -> np.ndarray:
    """Inject a temporary user, run HGT + user tower, return user_vec[new]."""
    data = copy.deepcopy(train_data)
    device = data["tire"].x.device

    new_user_idx = data["user"].num_nodes
    data["user"].num_nodes = new_user_idx + 1
    data["user"].node_id = torch.arange(new_user_idx + 1, device=device)

    # Expand learnable user embedding table with mean-init row.
    orig_emb = model.encoder.input_emb["user"]
    old_w = orig_emb.weight.data
    mean_row = old_w.mean(dim=0, keepdim=True)
    new_emb = nn.Embedding.from_pretrained(
        torch.cat([old_w, mean_row], dim=0), freeze=True
    ).to(device)
    model.encoder.input_emb["user"] = new_emb

    matching = torch.tensor(matching_tires, dtype=torch.long, device=device)
    n_match = matching.size(0)
    new_col = torch.full((n_match,), new_user_idx, dtype=torch.long, device=device)

    # Forward edges.
    fw_e = data["user", "reviews", "tire"].edge_index.to(device)
    new_fw = torch.stack([new_col, matching])
    data["user", "reviews", "tire"].edge_index = torch.cat([fw_e, new_fw], dim=1)
    fw_attr = data["user", "reviews", "tire"].edge_attr.to(device)
    new_attr = torch.full((n_match, 1), default_rating, device=device)
    data["user", "reviews", "tire"].edge_attr = torch.cat([fw_attr, new_attr], dim=0)

    # Reverse edges.
    rv_e = data["tire", "rev_by", "user"].edge_index.to(device)
    new_rv = torch.stack([matching, new_col])
    data["tire", "rev_by", "user"].edge_index = torch.cat([rv_e, new_rv], dim=1)
    rv_attr = data["tire", "rev_by", "user"].edge_attr.to(device)
    data["tire", "rev_by", "user"].edge_attr = torch.cat([rv_attr, new_attr], dim=0)

    # Build per-tire brand/size lookup on the augmented graph; new user
    # contributes one row to user_history (its preference matches).
    brand_idx, size_idx = build_tire_lookup(data)
    user_history = [[] for _ in range(new_user_idx + 1)]
    user_history[new_user_idx] = matching_tires

    cache = model.encode(data, brand_idx, size_idx, user_history)
    user_vec = cache["user_vec"][new_user_idx].cpu().numpy().astype("float32")

    # Restore original embedding.
    model.encoder.input_emb["user"] = orig_emb
    return user_vec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recommend tires (Two-Tower + FAISS).")
    p.add_argument("--user", type=str, required=True,
                   help='User index (int) or "new" for cold-start.')
    p.add_argument("--checkpoint", type=str,
                   default="outputs/checkpoints/two_tower_e20.pt")
    p.add_argument("--index-dir", type=str, default="outputs/index")
    p.add_argument("--graph", type=str, default="data/processed/hetero_graph.pt")
    p.add_argument("--k", type=int, default=10)

    # Cold-start preference filters.
    p.add_argument("--brand", type=str, default=None)
    p.add_argument("--size", type=str, default=None)
    p.add_argument("--budget-min", type=float, default=None)
    p.add_argument("--budget-max", type=float, default=None)
    p.add_argument("--min-treadwear", type=float, default=None)
    p.add_argument("--traction", type=str, default=None)
    p.add_argument("--temperature", type=str, default=None)

    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    # ── Load graph + mappings + tire df ──────────────────────────────
    graph_path = PROJECT_ROOT / args.graph
    payload = torch.load(graph_path, weights_only=False)
    data = payload["graph"].to(device)
    mappings = payload["mappings"]
    tire_df: pd.DataFrame = payload["tire_df"]
    idx_to_tire = {v: k for k, v in mappings["tire_map"].items()}

    # ── Load FAISS index ─────────────────────────────────────────────
    index_dir = (PROJECT_ROOT / args.index_dir).resolve()
    index = faiss.read_index(str(index_dir / "tire.faiss"))
    user_vecs = np.load(index_dir / "user_vectors.npy")  # warm-user cache

    # ── Existing user → FAISS lookup ─────────────────────────────────
    if args.user.lower() != "new":
        user_idx = int(args.user)
        if user_idx < 0 or user_idx >= user_vecs.shape[0]:
            print(f"Error: user index must be in [0, {user_vecs.shape[0] - 1}]")
            sys.exit(1)
        # Mask everything the user already reviewed in the train split.
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sampler = BPRSampler(
            data,
            rating_threshold=ckpt["args"]["rating_threshold"],
            seed=ckpt.get("split_seed", 0),
        )
        masked = set(sampler.user_reviewed_train[user_idx])

        results = faiss_search(index, user_vecs[user_idx], args.k, masked=masked)
        print(f"\nExisting user: {user_idx}")

    # ── Cold-start → inject user, recompute, FAISS ───────────────────
    else:
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
            print("Error: --user new requires at least one preference flag.")
            sys.exit(1)

        print("\nNew user preferences:")
        for k_, v_ in preferences.items():
            print(f"  {k_}: {v_}")

        matching = find_matching_tires(tire_df, mappings["tire_map"], preferences)
        print(f"\nMatching tires in catalog: {len(matching)} / {data['tire'].num_nodes}")
        if not matching:
            print("Error: no tires match — relax filters.")
            sys.exit(1)

        # Reconstruct the model and the train graph with the same seed.
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        train_args = ckpt["args"]
        sampler = BPRSampler(
            data,
            rating_threshold=train_args["rating_threshold"],
            seed=ckpt.get("split_seed", 0),
        )
        train_data = sampler.train_data

        model = TwoTowerRecommender.from_data(
            train_data,
            hidden_dim=train_args["hidden_dim"],
            num_layers=train_args["num_layers"],
            num_heads=train_args["num_heads"],
            out_dim=train_args["out_dim"],
            dropout=train_args["dropout"],
            init_temperature=train_args["init_temperature"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        user_vec = cold_start_user_vec(model, train_data, matching)
        results = faiss_search(index, user_vec, args.k, masked=None)
        print(f"\nCold-start user (matching tires injected: {len(matching)})")

    # ── Print top-K ──────────────────────────────────────────────────
    print(f"\nTop-{args.k} Recommended Tires:")
    print(f"{'Rank':<6} {'Score':<10} {'Tire'}")
    print("-" * 80)
    for rank, (t_idx, score) in enumerate(results, start=1):
        name = idx_to_tire.get(t_idx, f"tire_{t_idx}")
        if len(name) > 60:
            name = name[:57] + "..."
        print(f"{rank:<6} {score:<10.4f} {name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
