#!/usr/bin/env python3
"""
Regression checks for split integrity.

Verifies that:
  1. The train-only graph contains exactly the train review edges.
  2. Pseudo-label refresh and train_step encode only the train graph.
  3. BPR negatives never come from a user's train-reviewed tires.
  4. Contrastive triplets are sampled only from train-split interactions.

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

from scripts.inference import recommend_new_user
from src.models import TireRecommender
from src.training.sampler import BPRSampler
from src.training.trainer import Trainer


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

    model = TireRecommender.from_data(
        data,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_clusters=50,
    )
    assert "user" in model.encoder.input_emb

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

    original_encoder_forward = model.encoder.forward
    original_encode = model.encode
    encoder_seen: list[int] = []
    encode_seen: list[int] = []

    def checked_encoder_forward(graph):
        assert graph is sampler.train_data
        encoder_seen.append(id(graph))
        return original_encoder_forward(graph)

    def checked_encode(graph):
        assert graph is sampler.train_data
        encode_seen.append(id(graph))
        return original_encode(graph)

    model.encoder.forward = checked_encoder_forward
    model.encode = checked_encode

    trainer.refresh_pseudo_labels()
    stats = trainer.train_step(batch_size=64)

    assert encoder_seen, "refresh_pseudo_labels() did not use encoder on train_data"
    assert encode_seen, "train_step() did not encode train_data"
    assert torch.isfinite(torch.tensor(stats["loss"]))

    model.encoder.forward = original_encoder_forward
    model.encode = original_encode

    for _ in range(20):
        users, _, negs = sampler.sample(batch_size=128)
        for u, neg in zip(users.tolist(), negs.tolist()):
            assert neg not in sampler.user_reviewed_train[u]

    train_reviewed_tires = {
        int(t) for t in sampler.train_data["user", "reviews", "tire"].edge_index[1].tolist()
    }
    assert all(int(t) in train_reviewed_tires for t in sampler.val_tires.tolist())
    assert all(int(t) in train_reviewed_tires for t in sampler.test_tires.tolist())

    contrast = sampler.sample_contrast(batch_size=128)
    if contrast is not None:
        users, goods, bads = contrast
        for u, good, bad in zip(users.tolist(), goods.tolist(), bads.tolist()):
            assert good in sampler.user_positives[u]
            assert bad in sampler.user_disliked[u]
            assert good in sampler.user_reviewed_train[u]
            assert bad in sampler.user_reviewed_train[u]

    new_user_results = recommend_new_user(
        model,
        sampler.train_data,
        matching_tires=list(range(8)),
        k=5,
    )
    assert len(new_user_results) == 5

    print("Split integrity checks passed.")


if __name__ == "__main__":
    main()
