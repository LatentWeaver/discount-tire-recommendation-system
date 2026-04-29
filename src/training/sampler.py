"""
BPR + contrastive triplet sampler.

Two sampling streams sharing one user / item index:

  ``sample()`` — BPR triplets (u, t+, t-)
      t+  = a tire u rated at or above ``rating_threshold``
      t-  = a random tire absent from the user's train-split reviews

  ``sample_contrast()`` — contrastive triplets (u, t_good, t_disliked)
      t_good     = a tire u rated at or above ``rating_threshold``
      t_disliked = a tire u rated *below* ``rating_threshold``
      Only sampled from users who have at least one of each in train.

Random train / val / test split happens here so the trainer and
evaluator share the same partition. The split object also materialises
the train-only review graph used by the encoder, which prevents held-out
review edges from leaking into message passing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData


@dataclass
class ReviewEdgeSplit:
    """Canonical train / val / test split over review edges.

    The split reserves at least one review edge per tire in train before
    assigning the remaining review edges to validation/test. This avoids
    placing a tire entirely outside the train interaction graph.
    """

    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    train_data: HeteroData

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 0,
    ) -> "ReviewEdgeSplit":
        edge_index = data["user", "reviews", "tire"].edge_index
        n = edge_index.size(1)
        tires = edge_index[1].tolist()

        g = torch.Generator().manual_seed(seed)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        tire_to_edges: dict[int, list[int]] = {}
        for edge_idx, tire_idx in enumerate(tires):
            tire_to_edges.setdefault(tire_idx, []).append(edge_idx)

        reserved_train: list[int] = []
        candidate_holdout: list[int] = []
        for edge_ids in tire_to_edges.values():
            order = torch.randperm(len(edge_ids), generator=g).tolist()
            keep_train = edge_ids[order[0]]
            reserved_train.append(keep_train)
            candidate_holdout.extend(edge_ids[i] for i in order[1:])

        max_holdout = len(candidate_holdout)
        if n_test + n_val > max_holdout:
            raise ValueError(
                "Requested val/test split is too large to keep one train review "
                "edge per tire."
            )

        holdout_order = torch.randperm(max_holdout, generator=g).tolist()
        shuffled_holdout = [candidate_holdout[i] for i in holdout_order]

        test_idx = torch.tensor(shuffled_holdout[:n_test], dtype=torch.long)
        val_idx = torch.tensor(
            shuffled_holdout[n_test : n_test + n_val], dtype=torch.long
        )
        train_idx = torch.tensor(
            reserved_train + shuffled_holdout[n_test + n_val :],
            dtype=torch.long,
        )

        train_data = cls._build_train_graph(data, train_idx)
        return cls(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_data=train_data,
        )

    @staticmethod
    def _build_train_graph(data: HeteroData, train_idx: torch.Tensor) -> HeteroData:
        train_data = copy.deepcopy(data)

        fw_edge = ("user", "reviews", "tire")
        rv_edge = ("tire", "rev_by", "user")
        fw_store = train_data[fw_edge]
        idx = train_idx.to(fw_store.edge_index.device)

        fw_store.edge_index = fw_store.edge_index.index_select(1, idx)
        if getattr(fw_store, "edge_attr", None) is not None:
            fw_store.edge_attr = fw_store.edge_attr.index_select(0, idx)

        if rv_edge in train_data.edge_types:
            rv_store = train_data[rv_edge]
            rv_idx = train_idx.to(rv_store.edge_index.device)
            rv_store.edge_index = rv_store.edge_index.index_select(1, rv_idx)
            if getattr(rv_store, "edge_attr", None) is not None:
                rv_store.edge_attr = rv_store.edge_attr.index_select(0, rv_idx)

        return train_data


class BPRSampler:
    def __init__(
        self,
        data: HeteroData,
        rating_threshold: float = 4.0,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 0,
    ) -> None:
        edge_index = data["user", "reviews", "tire"].edge_index
        all_users = edge_index[0]
        all_tires = edge_index[1]
        edge_attr = getattr(data["user", "reviews", "tire"], "edge_attr", None)

        if edge_attr is None or rating_threshold is None:
            raise ValueError(
                "BPRSampler needs edge ratings (edge_attr) and a rating_threshold"
                " — bad reviews must be distinguishable from good ones."
            )

        self.num_users = data["user"].num_nodes
        self.num_tires = data["tire"].num_nodes
        self.split = ReviewEdgeSplit.from_data(
            data,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        self.train_data = self.split.train_data

        ratings = edge_attr.squeeze(-1)
        idx_device = all_users.device
        train_idx = self.split.train_idx.to(idx_device)
        val_idx = self.split.val_idx.to(idx_device)
        test_idx = self.split.test_idx.to(idx_device)

        train_users_all = all_users.index_select(0, train_idx)
        train_tires_all = all_tires.index_select(0, train_idx)
        train_ratings_all = ratings.index_select(0, train_idx)
        val_users_all = all_users.index_select(0, val_idx)
        val_tires_all = all_tires.index_select(0, val_idx)
        val_ratings_all = ratings.index_select(0, val_idx)
        test_users_all = all_users.index_select(0, test_idx)
        test_tires_all = all_tires.index_select(0, test_idx)
        test_ratings_all = ratings.index_select(0, test_idx)

        train_good_mask = train_ratings_all >= rating_threshold
        train_bad_mask = ~train_good_mask
        val_good_mask = val_ratings_all >= rating_threshold
        test_good_mask = test_ratings_all >= rating_threshold

        users = train_users_all[train_good_mask]
        tires = train_tires_all[train_good_mask]
        bad_users = train_users_all[train_bad_mask]
        bad_tires = train_tires_all[train_bad_mask]

        # Per-user reviewed items on the train split. BPR negatives must
        # avoid all of them, not just the train positives.
        self.user_reviewed_train: list[set[int]] = [
            set() for _ in range(self.num_users)
        ]
        for u, t in zip(train_users_all.tolist(), train_tires_all.tolist()):
            self.user_reviewed_train[u].add(t)

        # Per-user positives on the train split — used for BPR positives
        # and as train-time ground truth for evaluation masking.
        self.user_positives: list[set[int]] = [set() for _ in range(self.num_users)]
        for u, t in zip(users.tolist(), tires.tolist()):
            self.user_positives[u].add(t)

        # Per-user disliked tires — used for contrastive sampling.
        self.user_disliked: list[list[int]] = [[] for _ in range(self.num_users)]
        for u, t in zip(bad_users.tolist(), bad_tires.tolist()):
            self.user_disliked[u].append(t)

        # Per-user positives as a list (for fast random.choice during contrast).
        self.user_positives_list: list[list[int]] = [
            list(s) for s in self.user_positives
        ]

        # Pool of users who have BOTH ≥1 good and ≥1 bad review — only
        # these can produce a contrastive triplet.
        self.contrast_users: torch.Tensor = torch.tensor(
            [
                u
                for u in range(self.num_users)
                if self.user_positives_list[u] and self.user_disliked[u]
            ],
            dtype=torch.long,
        )

        self.train_users = users
        self.train_tires = tires
        self.val_users = val_users_all[val_good_mask]
        self.val_tires = val_tires_all[val_good_mask]
        self.test_users = test_users_all[test_good_mask]
        self.test_tires = test_tires_all[test_good_mask]

        self._gen = torch.Generator().manual_seed(seed + 1)

    # ──────────────────────────────────────────────────────────────────
    def sample_pos(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one mini-batch of (user, positive) pairs — no negatives.

        Used by the sampled-softmax loss, which gets its negatives from the
        other positives in the batch (in-batch negatives) and therefore
        does not need the per-row rejection loop.
        """
        n = self.train_users.size(0)
        idx = torch.randint(0, n, (batch_size,), generator=self._gen)
        return self.train_users[idx], self.train_tires[idx]

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw one mini-batch of BPR (user, positive, negative) triplets."""
        n = self.train_users.size(0)
        idx = torch.randint(0, n, (batch_size,), generator=self._gen)
        u = self.train_users[idx]
        pos = self.train_tires[idx]

        neg = torch.randint(0, self.num_tires, (batch_size,), generator=self._gen)
        for i in range(batch_size):
            ui = int(u[i])
            seen = self.user_reviewed_train[ui]
            while int(neg[i]) in seen:
                neg[i] = int(
                    torch.randint(0, self.num_tires, (1,), generator=self._gen).item()
                )

        return u, pos, neg

    # ──────────────────────────────────────────────────────────────────
    def sample_contrast(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Draw a mini-batch of contrastive (user, t_good, t_disliked) triplets.

        Returns ``None`` if no users in the dataset have both a good and a
        disliked review (the loss term must then be skipped).
        """
        pool = self.contrast_users
        if pool.numel() == 0:
            return None

        idx = torch.randint(0, pool.numel(), (batch_size,), generator=self._gen)
        u = pool[idx]

        good = torch.empty(batch_size, dtype=torch.long)
        bad = torch.empty(batch_size, dtype=torch.long)
        for i in range(batch_size):
            ui = int(u[i])
            goods = self.user_positives_list[ui]
            bads = self.user_disliked[ui]
            gi = int(torch.randint(0, len(goods), (1,), generator=self._gen).item())
            bi = int(torch.randint(0, len(bads), (1,), generator=self._gen).item())
            good[i] = goods[gi]
            bad[i] = bads[bi]

        return u, good, bad
