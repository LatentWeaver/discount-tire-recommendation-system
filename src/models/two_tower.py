"""
Two-Tower retrieval model on top of the HGT encoder.

  HGT(graph) → h_user, h_tire, h_brand, h_size
  ItemTower([h_tire, h_brand, h_size, structured_specs, (text_emb)])  → item_vec
  UserTower([h_user, mean(item_vec over user history)])                → user_vec
  score(u, t) = user_vec · item_vec    (ℓ2-normalised, dot == cosine)

If ``data["tire"].text_x`` is present (a precomputed sentence-transformer
embedding), it is fused into the item tower via a learnable projection
(Tier-1 augmentation C in new_instructions.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.models.hgt_encoder import HGTEncoder


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ItemTower(nn.Module):
    """
    Concatenates (h_tire, h_brand_of_tire, h_size_of_tire, structured_specs,
    optional text_proj) and projects to ``out_dim``.
    """

    def __init__(
        self,
        hgt_dim: int,
        spec_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float = 0.2,
        text_dim: int | None = None,
        text_proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.text_proj: nn.Module | None
        if text_dim is not None and text_dim > 0:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, text_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            in_dim = hgt_dim * 3 + spec_dim + text_proj_dim
        else:
            self.text_proj = None
            in_dim = hgt_dim * 3 + spec_dim

        self.mlp = _MLP(in_dim, hidden, out_dim, dropout)

    def forward(
        self,
        h_tire: torch.Tensor,
        h_brand_per_tire: torch.Tensor,
        h_size_per_tire: torch.Tensor,
        tire_specs: torch.Tensor,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [h_tire, h_brand_per_tire, h_size_per_tire, tire_specs]
        if self.text_proj is not None:
            if text_emb is None:
                raise ValueError("ItemTower has a text head but no text_emb was passed.")
            parts.append(self.text_proj(text_emb))
        x = torch.cat(parts, dim=-1)
        return F.normalize(self.mlp(x), p=2, dim=-1)


class UserTower(nn.Module):
    """Concatenates (h_user, history_pool) and projects to ``out_dim``."""

    def __init__(
        self,
        hgt_dim: int,
        item_vec_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = hgt_dim + item_vec_dim
        self.mlp = _MLP(in_dim, hidden, out_dim, dropout)

    def forward(self, h_user: torch.Tensor, history_pool: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_user, history_pool], dim=-1)
        return F.normalize(self.mlp(x), p=2, dim=-1)


class TwoTowerRecommender(nn.Module):
    """
    HGT encoder shared by both towers; item & user towers project to the
    retrieval space.

    A learnable temperature scales the logits for sampled-softmax.
    """

    def __init__(
        self,
        encoder: HGTEncoder,
        spec_dim: int,
        out_dim: int = 64,
        hidden: int = 128,
        dropout: float = 0.2,
        init_temperature: float = 0.07,
        text_dim: int | None = None,
        text_proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hgt_dim = encoder.hidden_dim
        self.uses_text = text_dim is not None and text_dim > 0

        self.item_tower = ItemTower(
            hgt_dim=hgt_dim,
            spec_dim=spec_dim,
            hidden=hidden,
            out_dim=out_dim,
            dropout=dropout,
            text_dim=text_dim,
            text_proj_dim=text_proj_dim,
        )
        self.user_tower = UserTower(
            hgt_dim=hgt_dim,
            item_vec_dim=out_dim,
            hidden=hidden,
            out_dim=out_dim,
            dropout=dropout,
        )

        self.log_temp = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(init_temperature))))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=1e-3, max=100.0)

    # ──────────────────────────────────────────────────────────
    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        out_dim: int = 64,
        dropout: float = 0.2,
        init_temperature: float = 0.07,
        text_proj_dim: int = 64,
    ) -> "TwoTowerRecommender":
        encoder = HGTEncoder.from_data(
            data,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        spec_dim = int(data["tire"].x.size(-1))
        text_x = getattr(data["tire"], "text_x", None)
        text_dim = int(text_x.size(-1)) if text_x is not None else None
        return cls(
            encoder=encoder,
            spec_dim=spec_dim,
            out_dim=out_dim,
            hidden=hidden_dim,
            dropout=dropout,
            init_temperature=init_temperature,
            text_dim=text_dim,
            text_proj_dim=text_proj_dim,
        )

    # ──────────────────────────────────────────────────────────
    def encode(
        self,
        data: HeteroData,
        tire_brand_idx: torch.Tensor,
        tire_size_idx: torch.Tensor,
        user_history: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        """One full forward over the train graph; returns cached vectors."""
        h_dict = self.encoder(data)
        h_user = h_dict["user"]
        h_tire = h_dict["tire"]
        h_brand = h_dict["brand"]
        h_size = h_dict["size"]

        h_brand_per_tire = h_brand[tire_brand_idx]
        h_size_per_tire = h_size[tire_size_idx]
        tire_specs = data["tire"].x
        text_emb = getattr(data["tire"], "text_x", None) if self.uses_text else None

        item_vec = self.item_tower(
            h_tire, h_brand_per_tire, h_size_per_tire, tire_specs, text_emb
        )

        history_pool = self._pool_history(item_vec, user_history)
        user_vec = self.user_tower(h_user, history_pool)

        return {
            "user_vec": user_vec,
            "item_vec": item_vec,
        }

    @staticmethod
    def _pool_history(
        item_vec: torch.Tensor,
        user_history: list[list[int]],
    ) -> torch.Tensor:
        n_users = len(user_history)
        d = item_vec.size(-1)
        out = item_vec.new_zeros((n_users, d))
        for u, items in enumerate(user_history):
            if items:
                out[u] = item_vec[items].mean(dim=0)
        return out

    # ──────────────────────────────────────────────────────────
    def score(
        self,
        cache: dict[str, torch.Tensor],
        users: torch.Tensor,
        tires: torch.Tensor,
    ) -> torch.Tensor:
        u = cache["user_vec"][users]
        t = cache["item_vec"][tires]
        return (u * t).sum(dim=-1) / self.temperature
