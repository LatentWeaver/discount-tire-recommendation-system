"""
Top-K ranking evaluation for the Two-Tower model.

Inputs are the cached ``user_vec`` / ``item_vec`` tensors produced once
per evaluation (one full HGT forward over the train-only review graph).

For each held-out (user, positive) pair:
  1. Rank every tire by ``user_vec[u] · item_vec[t]``.
  2. Mask all tires the user reviewed in the train split.
  3. Compute Recall@K, NDCG@K, HitRate@K against the held-out positive.

Masking is done with a single vectorised ``index_put_`` per batch — the
flat (row, tire) index is built once from ``user_reviewed_train`` and
sliced per batch on the GPU.
"""

from __future__ import annotations

import math

import torch


def _build_train_index(
    user_reviewed_train: list[set[int]] | list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten the per-user review sets into (user, tire) pair tensors."""
    flat_users: list[int] = []
    flat_tires: list[int] = []
    for u, seen in enumerate(user_reviewed_train):
        for t in seen:
            flat_users.append(u)
            flat_tires.append(int(t))
    if not flat_users:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(len(user_reviewed_train) + 1, dtype=torch.long, device=device)

    u_t = torch.tensor(flat_users, dtype=torch.long, device=device)
    t_t = torch.tensor(flat_tires, dtype=torch.long, device=device)
    # Sort by user so we can build a CSR-style offset and slice per batch.
    order = torch.argsort(u_t, stable=True)
    u_t = u_t[order]
    t_t = t_t[order]
    n_users = len(user_reviewed_train)
    counts = torch.zeros(n_users, dtype=torch.long, device=device)
    counts.scatter_add_(0, u_t, torch.ones_like(u_t))
    offsets = torch.zeros(n_users + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)
    return u_t, t_t, offsets


@torch.no_grad()
def evaluate(
    cache: dict[str, torch.Tensor],
    eval_users: torch.Tensor,
    eval_tires: torch.Tensor,
    user_reviewed_train: list[set[int]] | list[list[int]],
    ks: tuple[int, ...] = (10, 20, 50),
) -> dict[str, float]:
    user_vec = cache["user_vec"]
    item_vec = cache["item_vec"]
    device = item_vec.device

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({f"HitRate@{k}": 0.0 for k in ks})

    n = eval_users.size(0)
    if n == 0:
        return {k: 0.0 for k in sums}

    eval_users = eval_users.to(device)
    eval_tires = eval_tires.to(device)
    topk_max = max(ks)

    # Sorted-by-user CSR-ish index for fast train-mask lookup.
    flat_t, _, offsets = (None, None, None)
    _, sorted_tires, sorted_offsets = _build_train_index(user_reviewed_train, device)

    batch = 1024
    for start in range(0, n, batch):
        end = min(start + batch, n)
        u_b = eval_users[start:end]
        t_b = eval_tires[start:end]
        b = u_b.size(0)

        scores = user_vec[u_b] @ item_vec.t()  # (B, N_tire)

        # Build flat (row, tire) index from each user's offset slice, then
        # apply the mask in one shot.
        starts = sorted_offsets[u_b]            # (B,)
        ends = sorted_offsets[u_b + 1]          # (B,)
        lengths = ends - starts                 # (B,)
        total = int(lengths.sum().item())
        if total > 0:
            row_ids = torch.repeat_interleave(
                torch.arange(b, device=device), lengths
            )
            # within-row offset 0,1,...,len_i-1 for each row, then convert to a
            # flat index into ``sorted_tires``.
            cum_lengths = torch.cumsum(lengths, 0)              # (B,)
            row_end_per_k = cum_lengths.repeat_interleave(lengths)
            within_row = torch.arange(total, device=device) - (
                row_end_per_k - lengths.repeat_interleave(lengths)
            )
            row_starts_for_k = starts.repeat_interleave(lengths)
            col_ids = sorted_tires[row_starts_for_k + within_row]

            # Don't mask the held-out positive itself.
            held_out = t_b[row_ids]
            keep = col_ids != held_out
            scores[row_ids[keep], col_ids[keep]] = float("-inf")

        topk = torch.topk(scores, k=topk_max, dim=1).indices  # (B, K)

        # Vectorise rank computation: position of held-out tire in each row.
        match = (topk == t_b.unsqueeze(1))                    # (B, K)
        any_hit = match.any(dim=1)
        # rank = first index where match is True; sentinel if not found.
        rank = torch.where(
            any_hit,
            match.float().argmax(dim=1),
            torch.full_like(t_b, topk_max),
        )
        log_disc = 1.0 / torch.log2(rank.float() + 2.0)

        for k in ks:
            in_top = rank < k
            sums[f"Recall@{k}"] += float(in_top.sum().item())
            sums[f"HitRate@{k}"] += float(in_top.sum().item())
            sums[f"NDCG@{k}"] += float((in_top.float() * log_disc).sum().item())

    return {k: v / n for k, v in sums.items()}
