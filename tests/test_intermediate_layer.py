#!/usr/bin/env python3
"""
Smoke test: HGT encoder → IntermediateLayer → DeepCluster refresh →
auxiliary CE loss. Confirms shapes and that L_cluster is finite +
backprops cleanly.

Usage
-----
    uv run python tests/test_intermediate_layer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.losses.cluster_loss import deep_cluster_loss
from src.models import HGTEncoder, IntermediateLayer
from src.training.deep_cluster import refresh_pseudo_labels


def main() -> None:
    graph_path = PROJECT_ROOT / "data" / "processed" / "hetero_graph.pt"
    data = torch.load(graph_path, weights_only=False)

    hidden_dim = 128
    num_clusters = 50

    encoder = HGTEncoder.from_data(
        data, hidden_dim=hidden_dim, num_layers=2, num_heads=4
    )
    intermediate = IntermediateLayer(
        hidden_dim=hidden_dim,
        num_clusters=num_clusters,
        transform_dim=hidden_dim,
        dropout=0.1,
    )

    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_int = sum(p.numel() for p in intermediate.parameters() if p.requires_grad)
    print(f"Encoder params:      {n_enc:,}")
    print(f"Intermediate params: {n_int:,}")

    encoder.eval()
    intermediate.eval()
    with torch.no_grad():
        h_dict = encoder(data)
        out = intermediate(h_dict)

    print("\nHGT outputs:")
    for nt, h in h_dict.items():
        print(f"  {nt:>6s}: {tuple(h.shape)}")

    print("\nIntermediate outputs:")
    for k, v in out.items():
        print(f"  {k:<14s}: {tuple(v.shape)}")

    print("\nRefreshing pseudo-labels (PCA-64 → whiten → ℓ2 → k-means)...")
    pseudo = refresh_pseudo_labels(
        h_dict["tire"], num_clusters=num_clusters, pca_dim=64, seed=0
    )
    counts = torch.bincount(pseudo, minlength=num_clusters)
    print(
        f"  pseudo_labels: {tuple(pseudo.shape)} "
        f"(min={counts.min().item()}, max={counts.max().item()}, "
        f"mean={counts.float().mean().item():.1f})"
    )

    encoder.train()
    intermediate.train()
    h_dict = encoder(data)
    out = intermediate(h_dict)
    loss = deep_cluster_loss(
        out["cluster_logits"], pseudo, num_clusters=num_clusters
    )
    loss.backward()
    print(f"\nL_cluster = {loss.item():.4f}  (backward OK)")


if __name__ == "__main__":
    main()
