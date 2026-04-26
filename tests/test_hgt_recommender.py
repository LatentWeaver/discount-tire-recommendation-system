#!/usr/bin/env python3
"""
Smoke test for the pure-HGT recommender: load the graph, build the model,
run encode + score on a small batch, and check shapes / value ranges.

Usage
-----
    uv run python tests/test_hgt_recommender.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import HGTRecommender


def main() -> None:
    graph_path = PROJECT_ROOT / "data" / "processed" / "movielens_hetero_graph.pt"
    data = torch.load(graph_path, weights_only=False)["graph"]

    model = HGTRecommender.from_data(
        data, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.0,
        temperature=1.0,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    model.eval()
    with torch.no_grad():
        out = model.encode(data)
        for k, v in out.items():
            print(f"  {k:>10s}: {tuple(v.shape)}")

        n_user = out["h_user_t"].size(0)
        n_item = out["h_item_t"].size(0)
        users = torch.randint(0, n_user, (8,))
        items = torch.randint(0, n_item, (8,))
        scores = model.score(out, users, items)
        print(f"  scores: shape={tuple(scores.shape)} "
              f"min={scores.min().item():.4f} max={scores.max().item():.4f}")

        # With normalize=True, dot products are cosine similarities ∈ [-1, 1].
        assert scores.min().item() >= -1.0 - 1e-5
        assert scores.max().item() <= 1.0 + 1e-5
        print("  ✓ cosine score range OK")


if __name__ == "__main__":
    main()
