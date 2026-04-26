"""
Pure-HGT item recommender.

Encoder-only architecture: HGTEncoder produces user/item embeddings that
are L2-normalized and scored by dot product (= cosine similarity). No
intermediate transform, no cluster head, no fusion MLP.

Output dict keys are ``h_user_t`` and ``h_item_t``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.models.hgt_encoder import HGTEncoder


class HGTRecommender(nn.Module):
    def __init__(
        self,
        encoder: HGTEncoder,
        normalize: bool = True,
        temperature: float = 20.0,
        use_review_head: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        # Cosine logits live in [-1, 1] which saturates BPR's sigmoid; scale
        # them up so the margin can grow large enough for useful gradients.
        self.temperature = temperature
        self.review_head = (
            nn.Sequential(
                nn.Linear(encoder.hidden_dim * 4, encoder.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(encoder.hidden_dim, 1),
            )
            if use_review_head
            else None
        )

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        normalize: bool = True,
        temperature: float = 20.0,
        use_review_head: bool = True,
    ) -> "HGTRecommender":
        encoder = HGTEncoder.from_data(
            data,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        return cls(
            encoder,
            normalize=normalize,
            temperature=temperature,
            use_review_head=use_review_head,
        )

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h = self.encoder(data)
        h_user = h["user"]
        h_item = h["item"]
        if self.normalize:
            h_user = F.normalize(h_user, dim=-1)
            h_item = F.normalize(h_item, dim=-1)
        return {"h_user_t": h_user, "h_item_t": h_item}

    def score(
        self,
        out: dict[str, torch.Tensor],
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        dot = (out["h_user_t"][user_idx] * out["h_item_t"][item_idx]).sum(-1)
        return self.temperature * dot

    def review_logit(
        self,
        out: dict[str, torch.Tensor],
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict whether an observed review is liked vs. disliked."""
        if self.review_head is None:
            raise RuntimeError("review_head is disabled for this model.")
        h_user = out["h_user_t"][user_idx]
        h_item = out["h_item_t"][item_idx]
        pair = torch.cat(
            [h_user, h_item, h_user * h_item, torch.abs(h_user - h_item)],
            dim=-1,
        )
        return self.review_head(pair).squeeze(-1)
