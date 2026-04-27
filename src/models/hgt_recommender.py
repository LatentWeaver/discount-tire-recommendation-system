"""
Pure-HGT item recommender.

Encoder architecture: HGTEncoder produces user/item embeddings that are
L2-normalized and scored by a hybrid ranking decoder:
    temperature * dot(user, item) + pair MLP + user/item/global bias.

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
        num_users: int | None = None,
        num_items: int | None = None,
        use_rank_mlp: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        # Cosine logits live in [-1, 1] which saturates BPR's sigmoid; scale
        # them up so the margin can grow large enough for useful gradients.
        self.temperature = temperature
        self.rank_mlp = (
            nn.Sequential(
                nn.Linear(encoder.hidden_dim * 4, encoder.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(encoder.hidden_dim, 1),
            )
            if use_rank_mlp
            else None
        )
        if self.rank_mlp is not None:
            nn.init.zeros_(self.rank_mlp[-1].weight)
            nn.init.zeros_(self.rank_mlp[-1].bias)

        self.user_bias = (
            nn.Embedding(num_users, 1) if use_bias and num_users is not None else None
        )
        self.item_bias = (
            nn.Embedding(num_items, 1) if use_bias and num_items is not None else None
        )
        self.global_bias = nn.Parameter(torch.zeros(())) if use_bias else None
        if self.user_bias is not None:
            nn.init.zeros_(self.user_bias.weight)
        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias.weight)

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
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        normalize: bool = True,
        temperature: float = 20.0,
        use_review_head: bool = True,
        use_rank_mlp: bool = True,
        use_bias: bool = True,
        aggregate_layers: str = "mean",
    ) -> "HGTRecommender":
        encoder = HGTEncoder.from_data(
            data,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            aggregate_layers=aggregate_layers,
        )
        return cls(
            encoder,
            normalize=normalize,
            temperature=temperature,
            use_review_head=use_review_head,
            num_users=data["user"].num_nodes,
            num_items=data["item"].num_nodes,
            use_rank_mlp=use_rank_mlp,
            use_bias=use_bias,
        )

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h = self.encoder(data)
        h_user = h["user"]
        h_item = h["item"]
        if self.normalize:
            h_user = F.normalize(h_user, dim=-1)
            h_item = F.normalize(h_item, dim=-1)
        out = {"h_user_t": h_user, "h_item_t": h_item}
        user_node_id = getattr(data["user"], "node_id", None)
        item_node_id = getattr(data["item"], "node_id", None)
        if user_node_id is not None:
            out["user_node_id"] = user_node_id.to(h_user.device)
        if item_node_id is not None:
            out["item_node_id"] = item_node_id.to(h_item.device)
        return out

    def score(
        self,
        out: dict[str, torch.Tensor],
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        h_user = out["h_user_t"][user_idx]
        h_item = out["h_item_t"][item_idx]
        dot = (h_user * h_item).sum(-1)
        score = self.temperature * dot

        if self.rank_mlp is not None:
            score = score + self.rank_mlp(self._pair_features(h_user, h_item)).squeeze(
                -1
            )
        if self.user_bias is not None:
            bias_user_idx = (
                out["user_node_id"][user_idx] if "user_node_id" in out else user_idx
            )
            score = score + self.user_bias(bias_user_idx).squeeze(-1)
        if self.item_bias is not None:
            bias_item_idx = (
                out["item_node_id"][item_idx] if "item_node_id" in out else item_idx
            )
            score = score + self.item_bias(bias_item_idx).squeeze(-1)
        if self.global_bias is not None:
            score = score + self.global_bias
        return score

    @staticmethod
    def _pair_features(
        h_user: torch.Tensor, h_item: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [h_user, h_item, h_user * h_item, torch.abs(h_user - h_item)],
            dim=-1,
        )

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
        pair = self._pair_features(h_user, h_item)
        return self.review_head(pair).squeeze(-1)
