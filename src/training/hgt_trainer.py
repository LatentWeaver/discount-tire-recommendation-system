"""
Training loop for the pure-HGT recommender.

BPR-only objective. No cluster loss, no contrast loss, no pseudo-label
refresh. Encodes ``sampler.train_data`` so held-out edges never leak into
message passing.
"""

from __future__ import annotations

import torch

from src.losses.bpr import bpr_loss
from src.models.hgt_recommender import HGTRecommender
from src.training.evaluation import evaluate as _evaluate
from src.training.sampler import BPRSampler


class HGTTrainer:
    def __init__(
        self,
        model: HGTRecommender,
        sampler: BPRSampler,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.train_data = sampler.train_data
        self.sampler = sampler
        self.optimizer = optimizer

    def train_step(self, batch_size: int = 1024) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model.encode(self.train_data)
        device = out["h_user_t"].device

        u, pos, neg = self.sampler.sample(batch_size)
        score_pos = self.model.score(out, u.to(device), pos.to(device))
        score_neg = self.model.score(out, u.to(device), neg.to(device))
        loss = bpr_loss(score_pos, score_neg)

        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item()), "L_bpr": float(loss.item())}

    def evaluate(
        self, split: str = "val", ks: tuple[int, ...] = (10, 20, 50)
    ) -> dict[str, float]:
        if split == "val":
            users, items = self.sampler.val_users, self.sampler.val_items
        elif split == "test":
            users, items = self.sampler.test_users, self.sampler.test_items
        else:
            raise ValueError(f"Unknown split: {split!r}")

        return _evaluate(
            self.model,
            self.train_data,
            users,
            items,
            self.sampler.user_reviewed_train,
            ks=ks,
        )
