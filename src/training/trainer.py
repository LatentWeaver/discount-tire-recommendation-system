"""
Training loop coordinator for the Two-Tower model.

Per step:
  1. One full forward through the HGT encoder over the train-only review
     graph → ``user_vec``, ``item_vec``.
  2. Sample a mini-batch of positive (user, tire) pairs.
  3. Apply the chosen retrieval loss:
     - ``softmax`` (default) — sampled-softmax over **in-batch negatives**
       (the canonical Two-Tower objective).
     - ``bpr``               — pairwise BPR with one random negative.
     - ``bce``               — binary cross-entropy with one random negative.

CUDA optimizations baked in:
  - History pooling uses a precomputed flat (user, tire) pair tensor and
    GPU-side ``scatter_add_`` instead of a Python per-user loop.
  - When ``loss="softmax"`` we skip the per-row negative-rejection loop in
    the sampler entirely — in-batch negatives don't need it.
  - Optional bf16 ``autocast`` (CUDA only) via ``amp=True``.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.losses.bpr import bpr_loss
from src.models.two_tower import TwoTowerRecommender
from src.training.evaluation import evaluate as _evaluate
from src.training.sampler import BPRSampler


def build_tire_lookup(data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise per-tire brand/size node ids from the graph edges."""
    n_tires = data["tire"].num_nodes
    device = data["tire"].x.device

    brand_idx = torch.full((n_tires,), -1, dtype=torch.long, device=device)
    size_idx = torch.full((n_tires,), -1, dtype=torch.long, device=device)

    bt = data["tire", "belongs_to", "brand"].edge_index
    brand_idx[bt[0]] = bt[1]

    hs = data["tire", "has_spec", "size"].edge_index
    size_idx[hs[0]] = hs[1]

    if (brand_idx < 0).any() or (size_idx < 0).any():
        raise ValueError(
            "Some tires are missing a brand/size edge — graph_builder is expected"
            " to attach exactly one of each."
        )
    return brand_idx, size_idx


class TwoTowerTrainer:
    def __init__(
        self,
        model: TwoTowerRecommender,
        sampler: BPRSampler,
        optimizer: torch.optim.Optimizer,
        loss: str = "softmax",
        amp: bool = False,
        seed: int = 0,
    ) -> None:
        if loss not in {"softmax", "bpr", "bce"}:
            raise ValueError(f"Unknown loss '{loss}'. Use softmax / bpr / bce.")
        self.model = model
        self.sampler = sampler
        self.train_data = sampler.train_data
        self.optimizer = optimizer
        self.loss = loss
        self.seed = seed

        device = self.train_data["tire"].x.device
        self.device = device
        self.amp = bool(amp) and device.type == "cuda"

        # Per-tire brand/size node id (constant across training).
        self.tire_brand_idx, self.tire_size_idx = build_tire_lookup(self.train_data)

        # Flat (user_idx, tire_idx) pairs for vectorised history pooling.
        # Replaces the Python ``for u in range(N_user)`` loop in
        # TwoTowerRecommender._pool_history with a single scatter_add on GPU.
        flat_users: list[int] = []
        flat_tires: list[int] = []
        n_users = sampler.num_users
        for u, items in enumerate(sampler.user_positives_list):
            for t in items:
                flat_users.append(u)
                flat_tires.append(t)

        if flat_users:
            self.hist_user_idx = torch.tensor(flat_users, dtype=torch.long, device=device)
            self.hist_tire_idx = torch.tensor(flat_tires, dtype=torch.long, device=device)
        else:
            self.hist_user_idx = torch.zeros(0, dtype=torch.long, device=device)
            self.hist_tire_idx = torch.zeros(0, dtype=torch.long, device=device)

        counts = torch.zeros(n_users, dtype=torch.float32, device=device)
        if self.hist_user_idx.numel() > 0:
            counts.scatter_add_(
                0, self.hist_user_idx, torch.ones_like(self.hist_user_idx, dtype=torch.float32)
            )
        # clamp(min=1) so cold-start users (count=0) divide cleanly; their
        # numerator is also zero, so the pooled vector stays zero.
        self.hist_count = counts.clamp(min=1.0).unsqueeze(-1)
        self.hist_mask = (counts > 0)

        # AMP scaler is only meaningful for fp16; we use bf16, which doesn't
        # need a GradScaler. ``autocast(dtype=bf16)`` is enough.

    # ──────────────────────────────────────────────────────────────────
    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _pool_history_gpu(self, item_vec: torch.Tensor) -> torch.Tensor:
        """Mean-pool train-positive item vectors per user, fully on device."""
        n_users = self.sampler.num_users
        d = item_vec.size(-1)
        out = item_vec.new_zeros((n_users, d))
        if self.hist_user_idx.numel() == 0:
            return out
        contrib = item_vec.index_select(0, self.hist_tire_idx)  # (M, d)
        idx = self.hist_user_idx.unsqueeze(-1).expand(-1, d)
        out.scatter_add_(0, idx, contrib)
        return out / self.hist_count

    def _encode(self) -> dict[str, torch.Tensor]:
        """One full forward; same semantics as the model's ``encode`` but
        with the GPU-vectorised history pool."""
        h_dict = self.model.encoder(self.train_data)
        h_user = h_dict["user"]
        h_tire = h_dict["tire"]
        h_brand = h_dict["brand"]
        h_size = h_dict["size"]

        h_brand_per_tire = h_brand[self.tire_brand_idx]
        h_size_per_tire = h_size[self.tire_size_idx]
        tire_specs = self.train_data["tire"].x

        item_vec = self.model.item_tower(
            h_tire, h_brand_per_tire, h_size_per_tire, tire_specs
        )
        history_pool = self._pool_history_gpu(item_vec)
        user_vec = self.model.user_tower(h_user, history_pool)
        return {"user_vec": user_vec, "item_vec": item_vec}

    # ──────────────────────────────────────────────────────────────────
    def train_step(self, batch_size: int = 512) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            cache = self._encode()

            if self.loss == "softmax":
                u, pos = self.sampler.sample_pos(batch_size)
                u = u.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)

                user_vec = cache["user_vec"][u]
                pos_vec = cache["item_vec"][pos]
                logits = user_vec @ pos_vec.t() / self.model.temperature
                target = torch.arange(u.size(0), device=self.device)
                loss = F.cross_entropy(logits, target)

            elif self.loss == "bpr":
                u, pos, neg = self.sampler.sample(batch_size)
                u = u.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                neg = neg.to(self.device, non_blocking=True)
                score_pos = self.model.score(cache, u, pos)
                score_neg = self.model.score(cache, u, neg)
                loss = bpr_loss(score_pos, score_neg)

            else:  # bce
                u, pos, neg = self.sampler.sample(batch_size)
                u = u.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                neg = neg.to(self.device, non_blocking=True)
                score_pos = self.model.score(cache, u, pos)
                score_neg = self.model.score(cache, u, neg)
                scores = torch.cat([score_pos, score_neg])
                labels = torch.cat([
                    torch.ones_like(score_pos),
                    torch.zeros_like(score_neg),
                ])
                loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach().item()),
            "temp": float(self.model.temperature.item()),
        }

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(
        self,
        split: str = "val",
        ks: tuple[int, ...] = (10, 20, 50),
    ) -> dict[str, float]:
        if split == "val":
            users, tires = self.sampler.val_users, self.sampler.val_tires
        elif split == "test":
            users, tires = self.sampler.test_users, self.sampler.test_tires
        else:
            raise ValueError(f"Unknown split: {split!r}")

        self.model.eval()
        with self._autocast():
            cache = self._encode()
        # eval kept in fp32 to keep top-k stable
        cache = {k: v.float() for k, v in cache.items()}
        return _evaluate(
            cache=cache,
            eval_users=users,
            eval_tires=tires,
            user_reviewed_train=self.sampler.user_reviewed_train,
            ks=ks,
        )
