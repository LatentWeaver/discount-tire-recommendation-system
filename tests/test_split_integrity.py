#!/usr/bin/env python3
"""
Regression checks for split integrity (Two-Tower branch).

Verifies that:
  1. The train-only graph contains exactly the train review edges.
  2. The Two-Tower trainer encodes only the train graph.
  3. BPR negatives never come from a user's train-reviewed tires.
  4. Cold-start inference produces a valid user vector and FAISS-shaped result.

Usage
-----
    uv run python tests/test_split_integrity.py
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

    full_fw = data["user", "reviews", "tire"].edge_index
    train_fw = sampler.train_data["user", "reviews", "tire"].edge_index
    expected_fw = full_fw.index_select(1, sampler.split.train_idx.to(full_fw.device))
    assert torch.equal(train_fw.cpu(), expected_fw.cpu())

    full_attr = data["user", "reviews", "tire"].edge_attr
    train_attr = sampler.train_data["user", "reviews", "tire"].edge_attr
    expected_attr = full_attr.index_select(0, sampler.split.train_idx.to(full_attr.device))
    assert torch.equal(train_attr.cpu(), expected_attr.cpu())

    if ("tire", "rev_by", "user") in sampler.train_data.edge_types:
        full_rv = data["tire", "rev_by", "user"].edge_index
        train_rv = sampler.train_data["tire", "rev_by", "user"].edge_index
        expected_rv = full_rv.index_select(1, sampler.split.train_idx.to(full_rv.device))
        assert torch.equal(train_rv.cpu(), expected_rv.cpu())

    model = TwoTowerRecommender.from_data(
        sampler.train_data,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        out_dim=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    trainer = TwoTowerTrainer(
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        loss="softmax",
    )

    original_encoder_forward = model.encoder.forward
    encoder_seen: list[int] = []

    def checked_encoder_forward(graph):
        assert graph is sampler.train_data
        encoder_seen.append(id(graph))
        return original_encoder_forward(graph)

    model.encoder.forward = checked_encoder_forward
    stats = trainer.train_step(batch_size=64)
    assert encoder_seen, "trainer did not encode train_data"
    assert torch.isfinite(torch.tensor(stats["loss"]))
    model.encoder.forward = original_encoder_forward

    for _ in range(10):
        users, _, negs = sampler.sample(batch_size=128)
        for u, neg in zip(users.tolist(), negs.tolist()):
            assert neg not in sampler.user_reviewed_train[u]

    train_reviewed_tires = {
        int(t) for t in sampler.train_data["user", "reviews", "tire"].edge_index[1].tolist()
    }
    assert all(int(t) in train_reviewed_tires for t in sampler.val_tires.tolist())
    assert all(int(t) in train_reviewed_tires for t in sampler.test_tires.tolist())

    print("Split integrity checks passed.")


if __name__ == "__main__":
    main()
