"""
BPR + contrastive triplet sampler.

Two sampling streams sharing one user / item index:

  ``sample()`` — BPR triplets (u, t+, t-)
      t+  = a tire u rated at or above ``rating_threshold``
      t-  = a random tire u has not reviewed at all

  ``sample_contrast()`` — contrastive triplets (u, t_good, t_disliked)
      t_good     = a tire u rated at or above ``rating_threshold``
      t_disliked = a tire u rated *below* ``rating_threshold``
      Only sampled from users who have at least one of each.

Random train / val / test split happens here so the trainer and
evaluator share the same partition.
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData


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

        ratings = edge_attr.squeeze(-1)
        good_mask = ratings >= rating_threshold
        bad_mask = ~good_mask

        users = all_users[good_mask]
        tires = all_tires[good_mask]
        bad_users = all_users[bad_mask]
        bad_tires = all_tires[bad_mask]

        self.num_users = data["user"].num_nodes
        self.num_tires = data["tire"].num_nodes

        # Per-user positives — reject false negatives in BPR sampling
        # AND act as held-out ground truth for top-K eval.
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

        # Random train / val / test split on the (u, t) good-edge list.
        n = users.size(0)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_idx = perm[:n_test]
        val_idx = perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]

        self.train_users = users[train_idx]
        self.train_tires = tires[train_idx]
        self.val_users = users[val_idx]
        self.val_tires = tires[val_idx]
        self.test_users = users[test_idx]
        self.test_tires = tires[test_idx]

        self._gen = torch.Generator().manual_seed(seed + 1)

    # ──────────────────────────────────────────────────────────────────
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
            seen = self.user_positives[ui]
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
