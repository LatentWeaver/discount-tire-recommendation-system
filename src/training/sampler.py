"""
BPR + contrastive triplet sampler for user-item review graphs.

Expected graph schema:
    user --reviews--> item
    item --rev_by--> user
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData


@dataclass
class ReviewEdgeSplit:
    """Canonical train / val / test split over review edges.

    The split reserves at least one review edge per item in train before
    assigning the remaining review edges to validation/test. This avoids
    placing an item entirely outside the train interaction graph.
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
        edge_index = data["user", "reviews", "item"].edge_index
        n = edge_index.size(1)
        items = edge_index[1].tolist()

        g = torch.Generator().manual_seed(seed)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        item_to_edges: dict[int, list[int]] = {}
        for edge_idx, item_idx in enumerate(items):
            item_to_edges.setdefault(item_idx, []).append(edge_idx)

        reserved_train: list[int] = []
        candidate_holdout: list[int] = []
        for edge_ids in item_to_edges.values():
            order = torch.randperm(len(edge_ids), generator=g).tolist()
            keep_train = edge_ids[order[0]]
            reserved_train.append(keep_train)
            candidate_holdout.extend(edge_ids[i] for i in order[1:])

        max_holdout = len(candidate_holdout)
        if n_test + n_val > max_holdout:
            raise ValueError(
                "Requested val/test split is too large to keep one train review "
                "edge per item."
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

        return cls(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_data=cls._build_train_graph(data, train_idx),
        )

    @staticmethod
    def _build_train_graph(data: HeteroData, train_idx: torch.Tensor) -> HeteroData:
        train_data = copy.deepcopy(data)

        fw_edge = ("user", "reviews", "item")
        rv_edge = ("item", "rev_by", "user")
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
        edge_index = data["user", "reviews", "item"].edge_index
        all_users = edge_index[0]
        all_items = edge_index[1]
        edge_attr = getattr(data["user", "reviews", "item"], "edge_attr", None)

        if edge_attr is None or rating_threshold is None:
            raise ValueError(
                "BPRSampler needs edge ratings (edge_attr) and a rating_threshold"
                " so liked and disliked reviews are distinguishable."
            )

        self.num_users = data["user"].num_nodes
        self.num_items = data["item"].num_nodes
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
        train_items_all = all_items.index_select(0, train_idx)
        train_ratings_all = ratings.index_select(0, train_idx)
        val_users_all = all_users.index_select(0, val_idx)
        val_items_all = all_items.index_select(0, val_idx)
        val_ratings_all = ratings.index_select(0, val_idx)
        test_users_all = all_users.index_select(0, test_idx)
        test_items_all = all_items.index_select(0, test_idx)
        test_ratings_all = ratings.index_select(0, test_idx)

        train_good_mask = train_ratings_all >= rating_threshold
        train_bad_mask = ~train_good_mask
        val_good_mask = val_ratings_all >= rating_threshold
        test_good_mask = test_ratings_all >= rating_threshold

        users = train_users_all[train_good_mask]
        items = train_items_all[train_good_mask]
        bad_users = train_users_all[train_bad_mask]
        bad_items = train_items_all[train_bad_mask]

        self.user_reviewed_train: list[set[int]] = [
            set() for _ in range(self.num_users)
        ]
        for u, item in zip(train_users_all.tolist(), train_items_all.tolist()):
            self.user_reviewed_train[u].add(item)

        self.user_positives: list[set[int]] = [set() for _ in range(self.num_users)]
        for u, item in zip(users.tolist(), items.tolist()):
            self.user_positives[u].add(item)

        self.user_disliked: list[list[int]] = [[] for _ in range(self.num_users)]
        for u, item in zip(bad_users.tolist(), bad_items.tolist()):
            self.user_disliked[u].append(item)

        self.user_positives_list: list[list[int]] = [
            list(s) for s in self.user_positives
        ]
        self.contrast_users: torch.Tensor = torch.tensor(
            [
                u
                for u in range(self.num_users)
                if self.user_positives_list[u] and self.user_disliked[u]
            ],
            dtype=torch.long,
        )

        self.train_users = users
        self.train_items = items
        self.train_observed_users = train_users_all
        self.train_observed_items = train_items_all
        self.train_observed_labels = train_good_mask.float()
        self.val_users = val_users_all[val_good_mask]
        self.val_items = val_items_all[val_good_mask]
        self.val_observed_users = val_users_all
        self.val_observed_items = val_items_all
        self.val_observed_labels = val_good_mask.float()
        self.test_users = test_users_all[test_good_mask]
        self.test_items = test_items_all[test_good_mask]
        self.test_observed_users = test_users_all
        self.test_observed_items = test_items_all
        self.test_observed_labels = test_good_mask.float()

        self._gen = torch.Generator().manual_seed(seed + 1)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw one mini-batch of BPR (user, positive, negative) triplets."""
        n = self.train_users.size(0)
        idx = torch.randint(0, n, (batch_size,), generator=self._gen)
        users = self.train_users[idx]
        pos = self.train_items[idx]

        neg = torch.randint(0, self.num_items, (batch_size,), generator=self._gen)
        for i in range(batch_size):
            user_idx = int(users[i])
            seen = self.user_reviewed_train[user_idx]
            while int(neg[i]) in seen:
                neg[i] = int(
                    torch.randint(0, self.num_items, (1,), generator=self._gen).item()
                )

        return users, pos, neg

    def sample_observed(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw observed train reviews with binary liked/disliked labels."""
        n = self.train_observed_users.size(0)
        idx = torch.randint(0, n, (batch_size,), generator=self._gen)
        return (
            self.train_observed_users[idx],
            self.train_observed_items[idx],
            self.train_observed_labels[idx],
        )

    def sample_contrast(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Draw contrastive (user, liked_item, disliked_item) triplets."""
        pool = self.contrast_users
        if pool.numel() == 0:
            return None

        idx = torch.randint(0, pool.numel(), (batch_size,), generator=self._gen)
        users = pool[idx]

        good = torch.empty(batch_size, dtype=torch.long)
        bad = torch.empty(batch_size, dtype=torch.long)
        for i in range(batch_size):
            user_idx = int(users[i])
            goods = self.user_positives_list[user_idx]
            bads = self.user_disliked[user_idx]
            gi = int(torch.randint(0, len(goods), (1,), generator=self._gen).item())
            bi = int(torch.randint(0, len(bads), (1,), generator=self._gen).item())
            good[i] = goods[gi]
            bad[i] = bads[bi]

        return users, good, bad
