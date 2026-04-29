"""
Training loop coordinator for the Two-Tower model.

Per step:
  1. One full forward through the HGT encoder over the train-only review
     graph → ``user_vec``, ``item_vec``.
  2. Sample a mini-batch of positive (user, tire) pairs.
  3. Apply the chosen retrieval loss:
     - ``softmax`` (default) — sampled-softmax over **in-batch negatives**.
     - ``bpr``               — pairwise BPR with one random negative.
     - ``bce``               — binary cross-entropy with one random negative.
  4. (Optional) SGL-style graph contrastive: two augmented views of the
     train graph + InfoNCE on user/tire embeddings. Activated when
     ``ssl_lambda > 0``.

CUDA optimisations:
  - History pooling uses a precomputed flat (user, tire) pair tensor and
    GPU-side ``scatter_add_`` instead of a Python per-user loop.
  - ``loss="softmax"`` uses ``BPRSampler.sample_pos`` (no rejection loop).
  - Optional bf16 ``autocast`` (CUDA only) via ``amp=True``.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.losses.bpr import bpr_loss
from src.models.two_tower import TwoTowerRecommender
from src.training.augment import augment_view, info_nce
from src.training.evaluation import evaluate as _evaluate
from src.training.hard_negatives import build_buckets, sample_hard_negatives
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
        ssl_lambda: float = 0.0,
        ssl_edge_drop: float = 0.2,
        ssl_feat_drop: float = 0.1,
        ssl_tau: float = 0.5,
        ssl_sample_size: int = 1024,
        history_drop: float = 0.0,
        hard_neg_k: int = 0,
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

        self.ssl_lambda = float(ssl_lambda)
        self.ssl_edge_drop = float(ssl_edge_drop)
        self.ssl_feat_drop = float(ssl_feat_drop)
        self.ssl_tau = float(ssl_tau)
        self.ssl_sample_size = int(ssl_sample_size)
        self.history_drop = float(history_drop)
        self.hard_neg_k = int(hard_neg_k)
        self._neg_gen = torch.Generator(device=device.type if device.type != "mps" else "cpu").manual_seed(seed + 11)
        # Separate generator for SSL aug so it doesn't perturb sampler RNG.
        self._ssl_gen = (
            torch.Generator(device=device.type if device.type != "mps" else "cpu")
            .manual_seed(seed + 7)
        )

        # Per-tire brand/size node id (constant across training).
        self.tire_brand_idx, self.tire_size_idx = build_tire_lookup(self.train_data)

        # Flat (user_idx, tire_idx) pairs for vectorised history pooling.
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
        self.hist_count = counts.clamp(min=1.0).unsqueeze(-1)
        self.hist_mask = (counts > 0)

        # Hard-negative buckets (brand ∪ size). Built lazily so off-by-default
        # users don't pay the cost.
        if self.hard_neg_k > 0:
            self.bucket_offsets, self.bucket_indices = build_buckets(
                self.tire_brand_idx, self.tire_size_idx
            )
        else:
            self.bucket_offsets = self.bucket_indices = None

    # ──────────────────────────────────────────────────────────────────
    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _pool_history_gpu(self, item_vec: torch.Tensor) -> torch.Tensor:
        """Mean-pool train-positive item vectors per user (vectorised on GPU).

        When ``history_drop > 0`` and the model is in train mode, a fresh
        Bernoulli mask drops a fraction of (user, tire) pairs *before*
        pooling — counts are recomputed on the fly so the mean stays
        unbiased. Users that lose all of their history this step are
        treated like cold-start users (zero pool), which is exactly the
        regularisation we want for an eval-time empty-history regime.
        """
        n_users = self.sampler.num_users
        d = item_vec.size(-1)
        out = item_vec.new_zeros((n_users, d))
        if self.hist_user_idx.numel() == 0:
            return out

        contrib = item_vec.index_select(0, self.hist_tire_idx)  # (M, d)

        if self.history_drop > 0 and self.model.training:
            keep = (
                torch.rand(contrib.size(0), device=contrib.device)
                >= self.history_drop
            ).to(contrib.dtype)
            contrib = contrib * keep.unsqueeze(-1)
            kept_counts = torch.zeros(n_users, dtype=contrib.dtype, device=contrib.device)
            kept_counts.scatter_add_(0, self.hist_user_idx, keep)
            denom = kept_counts.clamp(min=1.0).unsqueeze(-1)
        else:
            denom = self.hist_count

        idx = self.hist_user_idx.unsqueeze(-1).expand(-1, d)
        out.scatter_add_(0, idx, contrib)
        return out / denom

    def _encode_graph(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        """Run encoder + towers on an arbitrary graph (train or augmented view)."""
        h_dict = self.model.encoder(graph)
        h_user = h_dict["user"]
        h_tire = h_dict["tire"]
        h_brand = h_dict["brand"]
        h_size = h_dict["size"]

        h_brand_per_tire = h_brand[self.tire_brand_idx]
        h_size_per_tire = h_size[self.tire_size_idx]
        tire_specs = graph["tire"].x
        text_emb = (
            getattr(graph["tire"], "text_x", None)
            if self.model.uses_text else None
        )

        item_vec = self.model.item_tower(
            h_tire, h_brand_per_tire, h_size_per_tire, tire_specs, text_emb
        )
        history_pool = self._pool_history_gpu(item_vec)
        user_vec = self.model.user_tower(h_user, history_pool)
        return {"user_vec": user_vec, "item_vec": item_vec}

    def _ssl_loss(self) -> torch.Tensor:
        """Two augmented views → InfoNCE on a sampled subset of users + tires."""
        v1 = augment_view(
            self.train_data,
            edge_drop=self.ssl_edge_drop,
            feat_drop=self.ssl_feat_drop,
            generator=self._ssl_gen,
        )
        v2 = augment_view(
            self.train_data,
            edge_drop=self.ssl_edge_drop,
            feat_drop=self.ssl_feat_drop,
            generator=self._ssl_gen,
        )
        c1 = self._encode_graph(v1)
        c2 = self._encode_graph(v2)

        n_user = c1["user_vec"].size(0)
        n_tire = c1["item_vec"].size(0)
        ss = self.ssl_sample_size

        u_idx = torch.randint(0, n_user, (min(ss, n_user),), device=self.device)
        t_idx = torch.randint(0, n_tire, (min(ss, n_tire),), device=self.device)

        loss_user = info_nce(c1["user_vec"][u_idx], c2["user_vec"][u_idx], self.ssl_tau)
        loss_tire = info_nce(c1["item_vec"][t_idx], c2["item_vec"][t_idx], self.ssl_tau)
        return 0.5 * (loss_user + loss_tire)

    # ──────────────────────────────────────────────────────────────────
    def train_step(self, batch_size: int = 512) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            cache = self._encode_graph(self.train_data)

            if self.loss == "softmax":
                u, pos = self.sampler.sample_pos(batch_size)
                u = u.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                user_vec = cache["user_vec"][u]
                pos_vec = cache["item_vec"][pos]
                temp = self.model.temperature
                logits = user_vec @ pos_vec.t() / temp                # (B, B)

                if self.hard_neg_k > 0:
                    hard = sample_hard_negatives(
                        pos, self.hard_neg_k,
                        self.bucket_offsets, self.bucket_indices,
                        num_tires=cache["item_vec"].size(0),
                        generator=self._neg_gen,
                    )                                                  # (B, k_hard)
                    hard_vec = cache["item_vec"][hard]                 # (B, k_hard, d)
                    hard_logits = (user_vec.unsqueeze(1) * hard_vec).sum(-1) / temp
                    logits = torch.cat([logits, hard_logits], dim=-1)  # (B, B + k_hard)

                target = torch.arange(u.size(0), device=self.device)
                loss_main = F.cross_entropy(logits, target)
            elif self.loss == "bpr":
                u, pos, neg = self.sampler.sample(batch_size)
                u = u.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                neg = neg.to(self.device, non_blocking=True)
                if self.hard_neg_k > 0:
                    hard = sample_hard_negatives(
                        pos, 1, self.bucket_offsets, self.bucket_indices,
                        num_tires=cache["item_vec"].size(0), generator=self._neg_gen,
                    ).squeeze(-1)
                    neg = hard
                loss_main = bpr_loss(
                    self.model.score(cache, u, pos),
                    self.model.score(cache, u, neg),
                )
            else:  # bce
                u, pos, neg = self.sampler.sample(batch_size)
                u = u.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                neg = neg.to(self.device, non_blocking=True)
                if self.hard_neg_k > 0:
                    hard = sample_hard_negatives(
                        pos, 1, self.bucket_offsets, self.bucket_indices,
                        num_tires=cache["item_vec"].size(0), generator=self._neg_gen,
                    ).squeeze(-1)
                    neg = hard
                score_pos = self.model.score(cache, u, pos)
                score_neg = self.model.score(cache, u, neg)
                scores = torch.cat([score_pos, score_neg])
                labels = torch.cat([
                    torch.ones_like(score_pos),
                    torch.zeros_like(score_neg),
                ])
                loss_main = F.binary_cross_entropy_with_logits(scores, labels)

            loss_ssl = torch.zeros((), device=self.device)
            if self.ssl_lambda > 0:
                loss_ssl = self._ssl_loss()

            loss = loss_main + self.ssl_lambda * loss_ssl

        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach().item()),
            "L_main": float(loss_main.detach().item()),
            "L_ssl": float(loss_ssl.detach().item()),
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
            cache = self._encode_graph(self.train_data)
        cache = {k: v.float() for k, v in cache.items()}
        return _evaluate(
            cache=cache,
            eval_users=users,
            eval_tires=tires,
            user_reviewed_train=self.sampler.user_reviewed_train,
            ks=ks,
        )
