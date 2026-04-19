"""
Intermediate Layer: Dual-Path Processing.

Wraps Path A (DeepCluster classifier head over h_tire) and Path B
(two-tower feature transform) into a single nn.Module so the training
loop can call one forward pass after the HGT encoder.

Outputs (dict):
    cluster_logits : (N_tire, K)  — fed to the auxiliary CE loss
    C_tire         : (N_tire, K)  — softmax of cluster_logits, fused into MLP
    h_user_t       : (N_user, D′) — Path B user projection
    h_tire_t       : (N_tire, D′) — Path B tire projection
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cluster_head import ClusterHead
from src.models.path_b import FeatureTransform


class IntermediateLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int,
        transform_dim: int | None = None,
        cluster_mlp_hidden: int | None = None,
        transform_mlp_hidden: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_head = ClusterHead(
            in_dim=hidden_dim,
            num_clusters=num_clusters,
            hidden_dim=cluster_mlp_hidden,
            dropout=dropout,
        )
        self.path_b = FeatureTransform(
            in_dim=hidden_dim,
            out_dim=transform_dim,
            hidden_dim=transform_mlp_hidden,
            dropout=dropout,
        )

    def forward(self, h_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h_user = h_dict["user"]
        h_tire = h_dict["tire"]

        cluster_logits = self.cluster_head(h_tire)
        c_tire = F.softmax(cluster_logits, dim=-1)
        h_user_t, h_tire_t = self.path_b(h_user, h_tire)

        return {
            "cluster_logits": cluster_logits,
            "C_tire": c_tire,
            "h_user_t": h_user_t,
            "h_tire_t": h_tire_t,
        }
