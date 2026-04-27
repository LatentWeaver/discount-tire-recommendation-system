"""
Pretraining loop for the HGT item recommender.

The objective combines:
  1. BPR ranking: push liked items above unreviewed items.
  2. Review classification: predict liked vs. disliked on observed reviews.

This is item-centric: the encoder learns reusable item representations from
ratings plus any metadata relations present in the graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.bpr import bpr_loss
from src.models.hgt_recommender import HGTRecommender
from src.training.evaluation import evaluate as _evaluate
from src.training.pyg_link_sampler import PyGLinkNeighborBatcher
from src.training.sampler import BPRSampler
from src.training.subgraph_sampler import HGTSubgraphSampler


class HGTPretrainer:
    def __init__(
        self,
        model: HGTRecommender,
        sampler: BPRSampler,
        optimizer: torch.optim.Optimizer,
        review_loss_weight: float = 0.5,
        subgraph_sampler: HGTSubgraphSampler | None = None,
        pyg_link_batcher: PyGLinkNeighborBatcher | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_data = sampler.train_data
        self.sampler = sampler
        self.optimizer = optimizer
        self.review_loss_weight = review_loss_weight
        self.subgraph_sampler = subgraph_sampler
        self.pyg_link_batcher = pyg_link_batcher
        self.device = device
        pos_rate = float(sampler.train_observed_labels.mean().item())
        neg_rate = 1.0 - pos_rate
        self.review_pos_weight = 0.5 / max(pos_rate, 1e-6)
        self.review_neg_weight = 0.5 / max(neg_rate, 1e-6)

    def train_step(self, batch_size: int = 1024) -> dict[str, float]:
        if self.pyg_link_batcher is not None:
            return self._train_pyg_link_step()
        if self.subgraph_sampler is not None:
            return self._train_subgraph_step(batch_size)

        self.model.train()
        self.optimizer.zero_grad()

        out = self.model.encode(self.train_data)
        device = out["h_user_t"].device

        u, pos, neg = self.sampler.sample(batch_size)
        score_pos = self.model.score(out, u.to(device), pos.to(device))
        score_neg = self.model.score(out, u.to(device), neg.to(device))
        loss_bpr = bpr_loss(score_pos, score_neg)

        obs_u, obs_item, obs_y = self.sampler.sample_observed(batch_size)
        logits = self.model.review_logit(out, obs_u.to(device), obs_item.to(device))
        labels = obs_y.to(device)
        review_weights = torch.where(
            labels > 0.5,
            torch.full_like(labels, self.review_pos_weight),
            torch.full_like(labels, self.review_neg_weight),
        )
        loss_review = F.binary_cross_entropy_with_logits(
            logits, labels, weight=review_weights
        )

        loss = loss_bpr + self.review_loss_weight * loss_review
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "L_bpr": float(loss_bpr.item()),
            "L_review": float(loss_review.item()),
        }

    def _train_pyg_link_step(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        device = self.device or next(self.model.parameters()).device
        batch_data = self.pyg_link_batcher.sample().to(device)
        out = self.model.encode(batch_data)

        edge_store = batch_data["user", "reviews", "item"]
        edge_label_index = edge_store.edge_label_index
        labels = edge_store.edge_label.float()
        logits = self.model.score(out, edge_label_index[0], edge_label_index[1])

        loss_link = self._grouped_bpr_loss(
            logits=logits,
            users=edge_label_index[0],
            labels=labels,
        )

        loss_link.backward()
        self.optimizer.step()

        return {
            "loss": float(loss_link.item()),
            "L_bpr": float(loss_link.item()),
            "L_review": 0.0,
        }

    @staticmethod
    def _grouped_bpr_loss(
        logits: torch.Tensor,
        users: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        pos_scores = logits[pos_mask]
        pos_users = users[pos_mask]
        neg_scores = logits[neg_mask]
        neg_users = users[neg_mask]

        losses: list[torch.Tensor] = []
        for user, pos_score in zip(pos_users.tolist(), pos_scores):
            user_neg_scores = neg_scores[neg_users == user]
            if user_neg_scores.numel() == 0:
                user_neg_scores = neg_scores
            if user_neg_scores.numel() > 0:
                losses.append(-F.logsigmoid(pos_score - user_neg_scores).mean())

        if losses:
            return torch.stack(losses).mean()
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _train_subgraph_step(self, batch_size: int = 1024) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        batch = self.subgraph_sampler.sample(batch_size)
        device = self.device or next(self.model.parameters()).device
        batch_data = batch.data.to(device)
        out = self.model.encode(batch_data)

        score_pos = self.model.score(
            out,
            batch.users.to(device),
            batch.pos_items.to(device),
        )
        score_neg = self.model.score(
            out,
            batch.users.to(device),
            batch.neg_items.to(device),
        )
        loss_bpr = bpr_loss(score_pos, score_neg)

        logits = self.model.review_logit(
            out,
            batch.observed_users.to(device),
            batch.observed_items.to(device),
        )
        labels = batch.observed_labels.to(device)
        review_weights = torch.where(
            labels > 0.5,
            torch.full_like(labels, self.review_pos_weight),
            torch.full_like(labels, self.review_neg_weight),
        )
        loss_review = F.binary_cross_entropy_with_logits(
            logits, labels, weight=review_weights
        )

        loss = loss_bpr + self.review_loss_weight * loss_review
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "L_bpr": float(loss_bpr.item()),
            "L_review": float(loss_review.item()),
        }

    @torch.no_grad()
    def review_metrics(self, split: str = "val") -> dict[str, float]:
        self.model.eval()
        device = self.device or next(self.model.parameters()).device
        out = self.model.encode(self.train_data.clone().to(device))

        if split == "val":
            users = self.sampler.val_observed_users
            items = self.sampler.val_observed_items
            labels = self.sampler.val_observed_labels
        elif split == "test":
            users = self.sampler.test_observed_users
            items = self.sampler.test_observed_items
            labels = self.sampler.test_observed_labels
        else:
            raise ValueError(f"Unknown split: {split!r}")

        logits = self.model.review_logit(out, users.to(device), items.to(device))
        preds = (torch.sigmoid(logits).cpu() >= 0.5).float()
        labels = labels.cpu()
        acc = float((preds == labels).float().mean().item())
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        pos_acc = (
            float((preds[pos_mask] == labels[pos_mask]).float().mean().item())
            if pos_mask.any()
            else 0.0
        )
        neg_acc = (
            float((preds[neg_mask] == labels[neg_mask]).float().mean().item())
            if neg_mask.any()
            else 0.0
        )
        return {
            "ReviewAcc": acc,
            "ReviewBalancedAcc": 0.5 * (pos_acc + neg_acc),
            "ReviewPosAcc": pos_acc,
            "ReviewNegAcc": neg_acc,
        }

    def review_accuracy(self, split: str = "val") -> float:
        return self.review_metrics(split=split)["ReviewAcc"]

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
