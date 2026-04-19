"""
Training loop coordinator.

Holds:
  - model + optimizer
  - BPRSampler (train / val / test split + negative sampling)
  - frozen pseudo-labels from the most recent k-means refresh

Exposes:
  - ``refresh_pseudo_labels()`` — call every N epochs
  - ``train_step(batch_size)`` — one optimizer step on L_BPR + λ·L_cluster
  - ``evaluate(...)`` — proxied through src.training.evaluation
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.losses.bpr import bpr_loss
from src.losses.cluster_loss import deep_cluster_loss
from src.models.recommender import TireRecommender
from src.training.deep_cluster import refresh_pseudo_labels
from src.training.evaluation import evaluate as _evaluate
from src.training.sampler import BPRSampler


class Trainer:
    def __init__(
        self,
        model: TireRecommender,
        data: HeteroData,
        sampler: BPRSampler,
        optimizer: torch.optim.Optimizer,
        cluster_lambda: float = 0.5,
        contrast_lambda: float = 0.3,
        contrast_batch_size: int | None = None,
        pca_dim: int | None = 64,
        num_clusters: int | None = None,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.data = data
        self.sampler = sampler
        self.optimizer = optimizer
        self.cluster_lambda = cluster_lambda
        self.contrast_lambda = contrast_lambda
        self.contrast_batch_size = contrast_batch_size
        self.pca_dim = pca_dim
        self.num_clusters = num_clusters or model.intermediate.num_clusters
        self.seed = seed

        self.pseudo_labels: torch.Tensor | None = None

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def refresh_pseudo_labels(self) -> torch.Tensor:
        """Snapshot h_tire and re-run k-means; freeze the resulting labels."""
        self.model.eval()
        h_dict = self.model.encoder(self.data)
        labels = refresh_pseudo_labels(
            h_dict["tire"],
            num_clusters=self.num_clusters,
            pca_dim=self.pca_dim,
            seed=self.seed,
        )
        self.pseudo_labels = labels.to(h_dict["tire"].device)
        return self.pseudo_labels

    # ──────────────────────────────────────────────────────────────────
    def train_step(self, batch_size: int = 1024) -> dict[str, float]:
        if self.pseudo_labels is None:
            self.refresh_pseudo_labels()

        self.model.train()
        self.optimizer.zero_grad()

        out = self.model.encode(self.data)
        device = out["h_user_t"].device

        u, pos, neg = self.sampler.sample(batch_size)
        score_pos = self.model.score(out, u.to(device), pos.to(device))
        score_neg = self.model.score(out, u.to(device), neg.to(device))
        l_bpr = bpr_loss(score_pos, score_neg)

        l_cluster = deep_cluster_loss(
            out["cluster_logits"],
            self.pseudo_labels,
            num_clusters=self.num_clusters,
        )

        l_contrast = torch.zeros((), device=device)
        contrast_batch = self.contrast_batch_size or batch_size
        contrast = self.sampler.sample_contrast(contrast_batch)
        if contrast is not None and self.contrast_lambda > 0:
            uc, t_good, t_bad = contrast
            score_good = self.model.score(
                out, uc.to(device), t_good.to(device)
            )
            score_bad = self.model.score(out, uc.to(device), t_bad.to(device))
            # Reuse BPR's pairwise margin loss — same math, different
            # semantics: "prefer a tire the user liked over one they owned
            # and disliked."
            l_contrast = bpr_loss(score_good, score_bad)

        loss = (
            l_bpr
            + self.cluster_lambda * l_cluster
            + self.contrast_lambda * l_contrast
        )
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "L_bpr": float(l_bpr.item()),
            "L_cluster": float(l_cluster.item()),
            "L_contrast": float(l_contrast.item()),
        }

    # ──────────────────────────────────────────────────────────────────
    def evaluate(
        self, split: str = "val", ks: tuple[int, ...] = (10, 20, 50)
    ) -> dict[str, float]:
        if split == "val":
            users, tires = self.sampler.val_users, self.sampler.val_tires
        elif split == "test":
            users, tires = self.sampler.test_users, self.sampler.test_tires
        else:
            raise ValueError(f"Unknown split: {split!r}")

        # Train positives only — held-out items must remain rankable.
        train_pos: list[set[int]] = [set() for _ in range(self.sampler.num_users)]
        for u_i, t_i in zip(
            self.sampler.train_users.tolist(), self.sampler.train_tires.tolist()
        ):
            train_pos[u_i].add(t_i)

        return _evaluate(self.model, self.data, users, tires, train_pos, ks=ks)
