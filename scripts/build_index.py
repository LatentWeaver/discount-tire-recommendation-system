#!/usr/bin/env python3
"""
Build a FAISS retrieval index from a trained Two-Tower checkpoint.

  1. Load graph + checkpoint.
  2. One forward pass through the HGT + ItemTower over the train graph.
  3. Build an IndexFlatIP over the ℓ2-normalised item vectors (exact, fine
     for the ~6k tire catalog; switch to IndexHNSWFlat for larger).
  4. Persist:
       outputs/index/tire.faiss
       outputs/index/tire_id_map.json
       outputs/index/item_vectors.npy
       outputs/index/user_vectors.npy   (warm-user lookup)

Usage
-----
    uv run python scripts/build_index.py --checkpoint outputs/checkpoints/two_tower_e20.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index from a Two-Tower ckpt.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--graph", type=str, default="data/processed/hetero_graph_vehicle.pt")
    p.add_argument("--out-dir", type=str, default="outputs/index")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    # ── Load graph & checkpoint ──────────────────────────────────────
    graph_path = PROJECT_ROOT / args.graph
    payload = torch.load(graph_path, weights_only=False)
    data = payload["graph"].to(device)
    mappings = payload["mappings"]

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    split_seed = ckpt.get("split_seed", 0)

    sampler = BPRSampler(
        data,
        rating_threshold=train_args["rating_threshold"],
        seed=split_seed,
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

    # ── Forward → cached vectors ─────────────────────────────────────
    tire_brand_idx, tire_size_idx = build_tire_lookup(train_data)
    with torch.no_grad():
        cache = model.encode(
            train_data,
            tire_brand_idx,
            tire_size_idx,
            sampler.user_positives_list,
        )
    item_vec = cache["item_vec"].cpu().numpy().astype("float32")
    user_vec = cache["user_vec"].cpu().numpy().astype("float32")

    # ── FAISS index (IndexFlatIP — vectors are ℓ2-normalised). ───────
    n_tire, dim = item_vec.shape
    index = faiss.IndexFlatIP(dim)
    index.add(item_vec)

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "tire.faiss"))
    np.save(out_dir / "item_vectors.npy", item_vec)
    np.save(out_dir / "user_vectors.npy", user_vec)

    idx_to_tire = {v: k for k, v in mappings["tire_map"].items()}
    with open(out_dir / "tire_id_map.json", "w") as f:
        json.dump({str(i): idx_to_tire[i] for i in range(n_tire)}, f, indent=2)

    print(f"Built FAISS index over {n_tire:,} tires (dim={dim}).")
    print(f"  → {out_dir / 'tire.faiss'}")
    print(f"  → {out_dir / 'item_vectors.npy'}")
    print(f"  → {out_dir / 'user_vectors.npy'}")
    print(f"  → {out_dir / 'tire_id_map.json'}")


if __name__ == "__main__":
    main()
