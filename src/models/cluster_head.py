"""
DeepCluster classifier head.

Two-layer MLP that maps tire embeddings ``h_tire`` (B, D) to cluster logits
(B, K). Trained with cross-entropy against pseudo-labels produced by the
periodic k-means refresh in ``src.training.deep_cluster``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClusterHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_clusters: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        h = hidden_dim or in_dim
        self.num_clusters = num_clusters
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_clusters),
        )

    def forward(self, h_tire: torch.Tensor) -> torch.Tensor:
        return self.net(h_tire)
