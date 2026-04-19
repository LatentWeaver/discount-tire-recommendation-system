"""
Fusion & Ranking MLP.

Concatenates the Path B user / tire projections with the Path A cluster
distribution and produces a scalar ranking score s(u, t). Used with the
BPR pairwise loss — output is a raw real-valued score, not bounded by a
sigmoid and not interpreted as a rating.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(
        self,
        user_dim: int,
        tire_dim: int,
        cluster_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = user_dim + tire_dim + cluster_dim

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        h_user: torch.Tensor,
        h_tire: torch.Tensor,
        c_tire: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h_user, h_tire, c_tire], dim=-1)
        return self.net(x).squeeze(-1)
