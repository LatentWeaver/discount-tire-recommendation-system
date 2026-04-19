"""
Top-K ranking evaluation.

For each (user, held-out positive) pair in the eval set:
  1. Score every tire in the catalog for that user (or a sampled subset).
  2. Mask tires the user has already reviewed in the train split.
  3. Compute Recall@K, NDCG@K, HitRate@K against the held-out positive.
"""

from __future__ import annotations

import math

import torch

from src.models.recommender import TireRecommender


@torch.no_grad()
def evaluate(
    model: TireRecommender,
    data,
    eval_users: torch.Tensor,
    eval_tires: torch.Tensor,
    user_positives_train: list[set[int]],
    ks: tuple[int, ...] = (10, 20, 50),
) -> dict[str, float]:
    model.eval()
    out = model.encode(data)
    num_tires = out["h_tire_t"].size(0)
    all_tires = torch.arange(num_tires, device=out["h_tire_t"].device)

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({f"HitRate@{k}": 0.0 for k in ks})

    n = eval_users.size(0)
    for i in range(n):
        u = int(eval_users[i])
        pos = int(eval_tires[i])

        users_rep = torch.full((num_tires,), u, dtype=torch.long, device=all_tires.device)
        scores = model.score(out, users_rep, all_tires)

        # Mask out the user's training-set positives so they don't count.
        for t_train in user_positives_train[u]:
            if t_train != pos:
                scores[t_train] = float("-inf")

        topk_max = max(ks)
        topk = torch.topk(scores, k=topk_max).indices.tolist()

        for k in ks:
            in_top = pos in topk[:k]
            sums[f"Recall@{k}"] += float(in_top)
            sums[f"HitRate@{k}"] += float(in_top)
            if in_top:
                rank = topk[:k].index(pos)
                sums[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)

    return {k: v / n for k, v in sums.items()}
