#!/usr/bin/env python3
"""
Two-Tower + FAISS inference.

In this dataset a "user" is a vehicle (user_id = "MAKE|Model"); year is not
in the graph and is accepted only for display.

Modes:
  1. Existing user (warm)   — FAISS top-K against cached user_vec.
                              Selected by --user <int> or --vehicle MAKE|Model.
  2. Cold-start by history  — supply --prev-tires "title1;title2;..."; the
                              script injects a temporary user node, hooks
                              `reviews` edges to those tires, recomputes the
                              user vector, then FAISS top-K. If --vehicle is
                              also given and matches an existing user, the
                              prev-tires augment that vehicle's history
                              instead of creating a new node.
  3. Cold-start by filters  — preference-match the catalog (brand / size /
                              min-* numerics) and use the matches as
                              injected history. Used when --prev-tires is
                              omitted.

Usage
-----
    # Existing vehicle:
    uv run python scripts/inference.py --vehicle "TOYOTA|Tundra" --k 10

    # Vehicle + the tires they already own (year accepted, not used):
    uv run python scripts/inference.py \\
        --vehicle "TOYOTA|Tundra" --year 2020 \\
        --prev-tires "BFGoodrich All Terrain T/A KO2;Michelin Defender LTX M/S" \\
        --k 10

    # Brand-new buyer, no vehicle on file, only owned-tire history:
    uv run python scripts/inference.py --user new \\
        --prev-tires "BFGoodrich All Terrain T/A KO2" --k 10

    # Brand-new buyer, only structured preferences:
    uv run python scripts/inference.py --user new \\
        --brand "Michelin,Continental" --min-rating 4.0
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import faiss  # must follow torch: faiss-before-torch segfaults macOS deepcopy of tensor storage

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


# Numeric preference filters that map directly to tire_df columns. Each
# entry is (CLI flag → DataFrame column) and applied as a >= threshold mask.
_MIN_FILTERS: dict[str, str] = {
    "min_rating": "average_rating",
    "min_tread_life": "sub_tread_life_mean",
    "min_wet_traction": "sub_wet_traction_mean",
    "min_dry_traction": "sub_dry_traction_mean",
    "min_cornering": "sub_cornering_steering_mean",
    "min_ride_noise": "sub_ride_noise_mean",
    "min_ride_comfort": "sub_ride_comfort_mean",
}


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
    for key, col in _MIN_FILTERS.items():
        threshold = preferences.get(key)
        if threshold is not None:
            mask &= tire_df[col].fillna(0) >= threshold

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


def resolve_tire_titles(
    titles: list[str],
    tire_map: dict[str, int],
) -> tuple[list[int], list[str]]:
    """Map free-form tire titles → tire indices.

    Tries exact match first, then case-insensitive substring match. Returns
    (resolved_indices, unresolved_inputs).
    """
    catalog_lower = {t.lower(): i for t, i in tire_map.items()}
    resolved: list[int] = []
    unresolved: list[str] = []
    for raw in titles:
        s = raw.strip()
        if not s:
            continue
        if s in tire_map:
            resolved.append(tire_map[s])
            continue
        low = s.lower()
        hit = catalog_lower.get(low)
        if hit is not None:
            resolved.append(hit)
            continue
        # Substring fallback — pick the shortest matching catalog title to
        # avoid runaway prefix collisions.
        candidates = [
            (t, i) for t_low, i in catalog_lower.items()
            for t in [next(k for k in tire_map if k.lower() == t_low)]
            if low in t_low
        ]
        if candidates:
            t, i = min(candidates, key=lambda ti: len(ti[0]))
            resolved.append(i)
        else:
            unresolved.append(raw)
    # De-dupe while preserving order.
    seen: set[int] = set()
    deduped = [x for x in resolved if not (x in seen or seen.add(x))]
    return deduped, unresolved


@torch.no_grad()
def cold_start_user_vec(
    model: TwoTowerRecommender,
    train_data,
    matching_tires: list[int],
    default_rating: float = 4.5,
    base_user_idx: int | None = None,
) -> tuple[np.ndarray, set[int]]:
    """Run HGT + user tower for a synthetic / augmented user.

    If ``base_user_idx`` is None, a brand-new user node is appended to the
    graph with `reviews` edges to ``matching_tires``. Otherwise, the edges
    are appended to the existing user's adjacency and the recomputed
    ``user_vec`` is read at ``base_user_idx``.

    Returns (user_vec, mask) where ``mask`` is the set of tire indices to
    exclude from the FAISS top-K (the user's known positives — newly
    supplied prev-tires plus any pre-existing train edges).
    """
    data = copy.deepcopy(train_data)
    device = data["tire"].x.device

    if base_user_idx is None:
        target_idx = data["user"].num_nodes
        data["user"].num_nodes = target_idx + 1
        data["user"].node_id = torch.arange(target_idx + 1, device=device)

        # Expand the learnable user embedding table with a mean-init row.
        orig_emb = model.encoder.input_emb["user"]
        old_w = orig_emb.weight.data
        mean_row = old_w.mean(dim=0, keepdim=True)
        new_emb = nn.Embedding.from_pretrained(
            torch.cat([old_w, mean_row], dim=0), freeze=True
        ).to(device)
        model.encoder.input_emb["user"] = new_emb
    else:
        target_idx = base_user_idx
        orig_emb = None  # nothing to restore

    matching = torch.tensor(matching_tires, dtype=torch.long, device=device)
    n_match = matching.size(0)
    new_col = torch.full((n_match,), target_idx, dtype=torch.long, device=device)

    # Forward edges.
    fw_store = data["user", "reviews", "tire"]
    fw_store.edge_index = torch.cat(
        [fw_store.edge_index.to(device), torch.stack([new_col, matching])], dim=1
    )
    new_attr = torch.full((n_match, 1), default_rating, device=device)
    fw_store.edge_attr = torch.cat([fw_store.edge_attr.to(device), new_attr], dim=0)

    # Reverse edges.
    rv_store = data["tire", "rev_by", "user"]
    rv_store.edge_index = torch.cat(
        [rv_store.edge_index.to(device), torch.stack([matching, new_col])], dim=1
    )
    rv_store.edge_attr = torch.cat([rv_store.edge_attr.to(device), new_attr], dim=0)

    # History pool input. For brand-new users, only the supplied tires.
    # For augmented existing users, union with their train positives.
    n_users = data["user"].num_nodes
    user_history = [[] for _ in range(n_users)]
    if base_user_idx is not None:
        train_fw = train_data["user", "reviews", "tire"].edge_index
        existing_pos = train_fw[1, train_fw[0] == target_idx].tolist()
        user_history[target_idx] = list({*existing_pos, *matching_tires})
    else:
        user_history[target_idx] = matching_tires

    brand_idx, size_idx = build_tire_lookup(data)
    cache = model.encode(data, brand_idx, size_idx, user_history)
    user_vec = cache["user_vec"][target_idx].cpu().numpy().astype("float32")

    if orig_emb is not None:
        model.encoder.input_emb["user"] = orig_emb

    return user_vec, set(user_history[target_idx])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recommend tires (Two-Tower + FAISS).")
    # Vehicle / user identity. Exactly one of (--user, --vehicle) is required.
    p.add_argument("--user", type=str, default=None,
                   help='User index (int) or "new" for cold-start. '
                        'Use --vehicle "MAKE|Model" instead when known.')
    p.add_argument("--vehicle", type=str, default=None,
                   help='Vehicle key "MAKE|Model" (e.g., "TOYOTA|Tundra"). '
                        'Resolved against the graph user_map.')
    p.add_argument("--year", type=int, default=None,
                   help="Vehicle year. Accepted for display only — the graph "
                        "has no year field, so this does not affect ranking.")
    p.add_argument("--prev-tires", type=str, default=None,
                   help='Semicolon-separated list of tire titles the buyer '
                        'already owns. Each is resolved to a catalog index '
                        '(case-insensitive substring match) and injected as '
                        'the user\'s history before scoring.')

    p.add_argument("--checkpoint", type=str,
                   default="outputs/checkpoints/two_tower_vehicle_small_e50.pt")
    p.add_argument("--index-dir", type=str, default="outputs/index")
    p.add_argument("--graph", type=str, default="data/processed/hetero_graph_vehicle.pt")
    p.add_argument("--k", type=int, default=10)

    # Cold-start preference filters (used only when --prev-tires is omitted).
    p.add_argument("--brand", type=str, default=None)
    p.add_argument("--size", type=str, default=None)
    p.add_argument("--min-rating", type=float, default=None,
                   help="Minimum average_rating (0-5).")
    p.add_argument("--min-tread-life", type=float, default=None)
    p.add_argument("--min-wet-traction", type=float, default=None)
    p.add_argument("--min-dry-traction", type=float, default=None)
    p.add_argument("--min-cornering", type=float, default=None)
    p.add_argument("--min-ride-noise", type=float, default=None)
    p.add_argument("--min-ride-comfort", type=float, default=None)

    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def resolve_user_arg(
    args: argparse.Namespace,
    user_map: dict[str, int],
) -> tuple[int | None, str]:
    """Return (warm user_idx or None, mode) from --user / --vehicle.

    mode ∈ {"warm-index", "warm-vehicle", "new", "augment-vehicle"}:
        - warm-index / warm-vehicle: pure FAISS lookup, no graph mutation.
        - new: cold-start with no base user.
        - augment-vehicle: existing vehicle's graph edges + extra prev-tires.
    """
    if args.vehicle:
        key = args.vehicle.strip()
        # case-insensitive lookup against MAKE|Model
        match = next(
            (v for k, v in user_map.items() if k.upper() == key.upper()),
            None,
        )
        if match is None:
            print(f"Error: --vehicle {key!r} not found in user_map. "
                  f"Available examples: {list(user_map)[:5]}")
            sys.exit(1)
        if args.prev_tires:
            return match, "augment-vehicle"
        return match, "warm-vehicle"

    if args.user is None:
        print("Error: provide --vehicle MAKE|Model, --user <int>, or --user new.")
        sys.exit(1)

    if args.user.lower() == "new":
        return None, "new"

    return int(args.user), "warm-index"


def _load_model_and_train_data(
    args: argparse.Namespace,
    data,
    review_df,
    device: torch.device,
):
    """Reconstruct the model + train_data using the checkpoint's hyperparams."""
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    sampler = BPRSampler(
        data,
        rating_threshold=train_args["rating_threshold"],
        seed=ckpt.get("split_seed", 0),
        review_df=review_df,
    )
    model = TwoTowerRecommender.from_data(
        sampler.train_data,
        hidden_dim=train_args["hidden_dim"],
        num_layers=train_args["num_layers"],
        num_heads=train_args["num_heads"],
        out_dim=train_args["out_dim"],
        dropout=train_args["dropout"],
        init_temperature=train_args["init_temperature"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, sampler


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    # ── Load graph + mappings + tire df ──────────────────────────────
    graph_path = PROJECT_ROOT / args.graph
    payload = torch.load(graph_path, weights_only=False)
    data = payload["graph"].to(device)
    mappings = payload["mappings"]
    tire_df: pd.DataFrame = payload["tire_df"]
    review_df = payload.get("review_df")
    idx_to_tire = {v: k for k, v in mappings["tire_map"].items()}

    # ── Load FAISS index ─────────────────────────────────────────────
    index_dir = (PROJECT_ROOT / args.index_dir).resolve()
    index = faiss.read_index(str(index_dir / "tire.faiss"))
    user_vecs = np.load(index_dir / "user_vectors.npy")  # warm-user cache

    # ── Resolve identity → mode ──────────────────────────────────────
    base_user, mode = resolve_user_arg(args, mappings["user_map"])
    if args.year is not None:
        print(f"\n(Year {args.year} accepted but not used — graph has no year.)")

    # Resolve prev-tires once (used by both new and augment modes).
    prev_idxs: list[int] = []
    if args.prev_tires:
        titles = [t for t in args.prev_tires.split(";") if t.strip()]
        prev_idxs, unresolved = resolve_tire_titles(titles, mappings["tire_map"])
        print(f"\nPrevious tires resolved: {len(prev_idxs)}/{len(titles)}")
        for i in prev_idxs:
            print(f"  → [{i}] {idx_to_tire.get(i)}")
        for u in unresolved:
            print(f"  ✗ no catalog match: {u!r}")
        if not prev_idxs:
            print("Error: no --prev-tires resolved. Check spelling.")
            sys.exit(1)

    # ── Warm: vehicle or numeric index, no prev-tires ────────────────
    if mode in {"warm-index", "warm-vehicle"}:
        user_idx = base_user
        if user_idx is None or user_idx < 0 or user_idx >= user_vecs.shape[0]:
            print(f"Error: user index must be in [0, {user_vecs.shape[0] - 1}]")
            sys.exit(1)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sampler = BPRSampler(
            data,
            rating_threshold=ckpt["args"]["rating_threshold"],
            seed=ckpt.get("split_seed", 0),
            review_df=review_df,
        )
        masked = set(sampler.user_reviewed_train[user_idx])
        results = faiss_search(index, user_vecs[user_idx], args.k, masked=masked)
        label = (
            f"vehicle {args.vehicle!r} (idx={user_idx})"
            if mode == "warm-vehicle" else f"user {user_idx}"
        )
        print(f"\nExisting {label}")

    # ── Augment existing vehicle with newly supplied prev-tires ──────
    elif mode == "augment-vehicle":
        model, sampler = _load_model_and_train_data(args, data, review_df, device)
        user_vec, masked = cold_start_user_vec(
            model, sampler.train_data, prev_idxs, base_user_idx=base_user
        )
        # Also mask everything this vehicle reviewed in train.
        masked |= set(sampler.user_reviewed_train[base_user])
        results = faiss_search(index, user_vec, args.k, masked=masked)
        print(f"\nAugmented vehicle {args.vehicle!r} "
              f"(idx={base_user}, +{len(prev_idxs)} prev-tires)")

    # ── Cold-start: brand-new user (prev-tires or filters) ───────────
    else:  # mode == "new"
        if prev_idxs:
            history = prev_idxs
            print(f"\nNew user — history from --prev-tires ({len(history)} tires).")
        else:
            preferences: dict = {}
            if args.brand:
                preferences["brands"] = [b.strip() for b in args.brand.split(",")]
            if args.size:
                preferences["size"] = args.size
            for key in _MIN_FILTERS:
                val = getattr(args, key)
                if val is not None:
                    preferences[key] = val
            if not preferences:
                print("Error: --user new requires --prev-tires or a preference flag.")
                sys.exit(1)
            print("\nNew user preferences:")
            for k_, v_ in preferences.items():
                print(f"  {k_}: {v_}")
            history = find_matching_tires(tire_df, mappings["tire_map"], preferences)
            print(f"\nMatching tires in catalog: {len(history)} / {data['tire'].num_nodes}")
            if not history:
                print("Error: no tires match — relax filters.")
                sys.exit(1)

        model, sampler = _load_model_and_train_data(args, data, review_df, device)
        user_vec, masked = cold_start_user_vec(
            model, sampler.train_data, history, base_user_idx=None
        )
        results = faiss_search(index, user_vec, args.k, masked=masked)
        print(f"\nCold-start user (history injected: {len(history)})")

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
