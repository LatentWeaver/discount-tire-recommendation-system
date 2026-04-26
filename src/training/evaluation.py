"""
Top-K ranking evaluation.

For each (user, held-out positive) pair in the eval set:
  1. Score every tire in the catalog for that user (or a sampled subset).
  2. Mask tires the user has already reviewed in the train split.
  3. Compute Recall@K, NDCG@K, HitRate@K against the held-out positive.
"""

from __future__ import annotations

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
    batch_size: int = 128,
) -> dict[str, float]:
    model.eval()
    out = model.encode(data)
    num_tires = out["h_tire_t"].size(0)
    device = out["h_tire_t"].device
    topk_max = min(max(ks), num_tires)

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({f"HitRate@{k}": 0.0 for k in ks})

    n = eval_users.size(0)
    if n == 0:
        return {k: 0.0 for k in sums}

    h_tire = out["h_tire_t"]
    c_tire = out["C_tire"]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_users_cpu = eval_users[start:end].cpu()
        batch_tires_cpu = eval_tires[start:end].cpu()
        batch_users = batch_users_cpu.to(device)

        batch_size_actual = batch_users.size(0)
        h_user = out["h_user_t"][batch_users]
        h_user = h_user[:, None, :].expand(batch_size_actual, num_tires, -1)
        h_tire_batch = h_tire[None, :, :].expand(batch_size_actual, num_tires, -1)
        c_tire_batch = c_tire[None, :, :].expand(batch_size_actual, num_tires, -1)

        scores = model.fusion(
            h_user.reshape(-1, h_user.size(-1)),
            h_tire_batch.reshape(-1, h_tire_batch.size(-1)),
            c_tire_batch.reshape(-1, c_tire_batch.size(-1)),
        ).view(batch_size_actual, num_tires)

        # Mask out the user's training-set positives so they don't count.
        for row, (u, pos) in enumerate(
            zip(batch_users_cpu.tolist(), batch_tires_cpu.tolist())
        ):
            mask_items = [t for t in user_positives_train[u] if t != pos]
            if mask_items:
                scores[row, torch.tensor(mask_items, device=device)] = float("-inf")

        topk = torch.topk(scores, k=topk_max, dim=1).indices.cpu()
        targets = batch_tires_cpu.unsqueeze(1)
        matches = topk.eq(targets)

        for k in ks:
            kk = min(k, topk_max)
            hits = matches[:, :kk]
            hit_any = hits.any(dim=1)
            hit_count = float(hit_any.float().sum().item())
            sums[f"Recall@{k}"] += hit_count
            sums[f"HitRate@{k}"] += hit_count

            if hit_any.any():
                hit_rows, hit_cols = hits.nonzero(as_tuple=True)
                first_rank = torch.full((hits.size(0),), -1, dtype=torch.long)
                first_rank[hit_rows] = hit_cols
                valid_rank = first_rank[first_rank >= 0].float()
                ndcg = torch.reciprocal(torch.log2(valid_rank + 2.0)).sum()
                sums[f"NDCG@{k}"] += float(ndcg.item())

    return {k: v / n for k, v in sums.items()}
