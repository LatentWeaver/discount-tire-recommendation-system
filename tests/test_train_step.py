#!/usr/bin/env python3
"""
Smoke test: end-to-end Two-Tower pipeline.

  Build TwoTowerRecommender → BPRSampler → TwoTowerTrainer
  → 3 train_steps → 1 mini eval. Verifies the retrieval loss flows
  through the full graph and metrics are computable on the val split.

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

from src.models import TwoTowerRecommender
from src.training.sampler import BPRSampler
from src.training.trainer import TwoTowerTrainer


def main() -> None:
    torch.manual_seed(0)

    graph_path = PROJECT_ROOT / "data" / "processed" / "hetero_graph.pt"
    data = torch.load(graph_path, weights_only=False)["graph"]

    sampler = BPRSampler(data, rating_threshold=4.0, seed=0)
    print(
        f"Splits — train: {sampler.train_users.size(0):,}, "
        f"val: {sampler.val_users.size(0):,}, "
        f"test: {sampler.test_users.size(0):,}"
    )

    model = TwoTowerRecommender.from_data(
        sampler.train_data,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        out_dim=64,
        dropout=0.2,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Two-Tower params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    trainer = TwoTowerTrainer(
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        loss="softmax",
    )

    print("\nRunning 3 train steps (sampled-softmax)...")
    for step in range(1, 4):
        stats = trainer.train_step(batch_size=256)
        print(f"  step {step}: loss={stats['loss']:.4f} temp={stats['temp']:.3f}")

    print("\nMini-eval on val (first 200 samples for speed)...")
    sampler.val_users = sampler.val_users[:200]
    sampler.val_tires = sampler.val_tires[:200]
    metrics = trainer.evaluate(split="val", ks=(10, 20, 50))
    for k, v in metrics.items():
        print(f"  {k:<14s} {v:.4f}")


if __name__ == "__main__":
    main()
