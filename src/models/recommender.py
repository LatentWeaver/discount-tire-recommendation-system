"""
Top-level tire recommender.

Wires HGT encoder → IntermediateLayer → FusionMLP. Separates:
  - ``encode(data)``: one full-graph forward pass; returns a dict of
      tensors keyed by [h_user_t, h_tire_t, C_tire, cluster_logits].
  - ``score(out, user_idx, tire_idx)``: cheap gather + fusion on a batch.

This split lets one graph forward serve many (user, pos, neg) samples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.models.fusion import FusionMLP
from src.models.hgt_encoder import HGTEncoder
from src.models.intermediate import IntermediateLayer


class TireRecommender(nn.Module):
    def __init__(
        self,
        encoder: HGTEncoder,
        intermediate: IntermediateLayer,
        fusion: FusionMLP,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.intermediate = intermediate
        self.fusion = fusion

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_clusters: int = 50,
        transform_dim: int | None = None,
        fusion_hidden_dim: int = 128,
        fusion_num_layers: int = 2,
        dropout: float = 0.1,
    ) -> "TireRecommender":
        encoder = HGTEncoder.from_data(
            data,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        transform_dim = transform_dim or hidden_dim
        intermediate = IntermediateLayer(
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            transform_dim=transform_dim,
            dropout=dropout,
        )
        fusion = FusionMLP(
            user_dim=transform_dim,
            tire_dim=transform_dim,
            cluster_dim=num_clusters,
            hidden_dim=fusion_hidden_dim,
            num_layers=fusion_num_layers,
            dropout=dropout,
        )
        return cls(encoder, intermediate, fusion)

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h_dict = self.encoder(data)
        return self.intermediate(h_dict)

    def score(
        self,
        out: dict[str, torch.Tensor],
        user_idx: torch.Tensor,
        tire_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(
            out["h_user_t"][user_idx],
            out["h_tire_t"][tire_idx],
            out["C_tire"][tire_idx],
        )
