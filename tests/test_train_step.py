#!/usr/bin/env python3
"""
Smoke test: end-to-end pipeline.

  Build TireRecommender → BPRSampler → Trainer → 1 refresh + 3 train_steps
  → 1 mini eval. Verifies BPR + cluster loss flow through the full graph
  and metrics are computable on the val split.

Usage
-----
    uv run python tests/test_train_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TireRecommender
from src.training.sampler import BPRSampler
from src.training.trainer import Trainer


def main() -> None:
    torch.manual_seed(0)

    graph_path = PROJECT_ROOT / "data" / "processed" / "hetero_graph.pt"
    data = torch.load(graph_path, weights_only=False)["graph"]

    model = TireRecommender.from_data(
        data,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_clusters=50,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Recommender params: {n_params:,}")

    sampler = BPRSampler(data, rating_threshold=4.0, seed=0)
    print(
        f"Splits — train: {sampler.train_users.size(0):,}, "
        f"val: {sampler.val_users.size(0):,}, "
        f"test: {sampler.test_users.size(0):,}"
    )
    print(
        f"Contrast pool — {sampler.contrast_users.numel():,} users "
        f"have both good AND disliked reviews"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        data=data,
        sampler=sampler,
        optimizer=optimizer,
        cluster_lambda=0.5,
        contrast_lambda=0.3,
        pca_dim=64,
        num_clusters=50,
        seed=0,
    )

    print("\nRefreshing pseudo-labels…")
    labels = trainer.refresh_pseudo_labels()
    counts = torch.bincount(labels, minlength=50)
    print(
        f"  cluster sizes — min={counts.min().item()}, "
        f"max={counts.max().item()}, mean={counts.float().mean().item():.1f}"
    )

    print("\nRunning 3 train steps…")
    for step in range(1, 4):
        stats = trainer.train_step(batch_size=512)
        print(
            f"  step {step}: loss={stats['loss']:.4f} "
            f"BPR={stats['L_bpr']:.4f} "
            f"cluster={stats['L_cluster']:.4f} "
            f"contrast={stats['L_contrast']:.4f}"
        )

    print("\nMini-eval on val (first 200 samples for speed)…")
    sampler.val_users = sampler.val_users[:200]
    sampler.val_tires = sampler.val_tires[:200]
    metrics = trainer.evaluate(split="val", ks=(10, 20, 50))
    for k, v in metrics.items():
        print(f"  {k:<14s} {v:.4f}")


if __name__ == "__main__":
    main()
