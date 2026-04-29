#!/usr/bin/env python3
"""
Evaluate a trained Two-Tower checkpoint on the val/test splits.

Loads a checkpoint, reconstructs the train graph with the same split
seed, runs one HGT + tower forward, and reports Recall@K / NDCG@K /
HitRate@K with all train-reviewed tires masked per user.

Usage
-----
    uv run python scripts/evaluate.py --checkpoint outputs/checkpoints/two_tower_e20.pt
    uv run python scripts/evaluate.py --checkpoint ... --ks 5,10,20,50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TwoTowerRecommender
from src.training.evaluation import evaluate as evaluate_ranking
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
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--graph", type=str, default="data/processed/hetero_graph.pt")
    p.add_argument("--ks", type=str, default="10,20,50")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ks = tuple(sorted({int(k.strip()) for k in args.ks.split(",") if k.strip()}))

    device = pick_device(args.device)
    payload = torch.load(PROJECT_ROOT / args.graph, weights_only=False)
    data = payload["graph"].to(device)

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

    brand_idx, size_idx = build_tire_lookup(train_data)
    with torch.no_grad():
        cache = model.encode(
            train_data, brand_idx, size_idx, sampler.user_positives_list
        )

    for split, users, tires in [
        ("val", sampler.val_users, sampler.val_tires),
        ("test", sampler.test_users, sampler.test_tires),
    ]:
        print(f"\n{split.upper()} ({users.size(0):,} pairs)")
        metrics = evaluate_ranking(
            cache=cache,
            eval_users=users,
            eval_tires=tires,
            user_reviewed_train=sampler.user_reviewed_train,
            ks=ks,
        )
        for k, v in metrics.items():
            print(f"  {k:<14s} {v:.4f}")


if __name__ == "__main__":
    main()
