"""
Path B: feature transformation.

Two-tower MLP that projects user / tire embeddings from the HGT encoder
into a matching space prepared for the downstream Fusion MLP. Separate
weights per tower because user and tire roles are asymmetric.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(d_in: int, d_h: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_h),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_h, d_out),
        nn.LayerNorm(d_out),
    )


class FeatureTransform(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.out_dim = out_dim
        self.user_mlp = _mlp(in_dim, hidden_dim, out_dim, dropout)
        self.tire_mlp = _mlp(in_dim, hidden_dim, out_dim, dropout)

    def forward(
        self, h_user: torch.Tensor, h_tire: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mlp(h_user), self.tire_mlp(h_tire)
