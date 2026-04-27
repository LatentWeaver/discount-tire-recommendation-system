#!/usr/bin/env python3
"""
Memory-safe evaluation for HGT checkpoints.

This script avoids full-graph HGT encoding. It computes user/item embeddings
with mini-batch heterogeneous subgraph inference, then evaluates full-catalog
top-K ranking with batched scoring.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hgt import pick_device
from src.models.hgt_recommender import HGTRecommender
from src.training.sampler import BPRSampler
from src.training.subgraph_sampler import HGTSubgraphSampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an HGT checkpoint.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/hgt_pretrained.pt",
    )
    p.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Override graph path stored in checkpoint.",
    )
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--rating-threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--embedding-batch-size", type=int, default=256)
    p.add_argument("--score-batch-size", type=int, default=256)
    p.add_argument("--item-chunk-size", type=int, default=2048)
    p.add_argument(
        "--max-eval-examples",
        type=int,
        default=None,
        help="Evaluate only N held-out positive examples for quick checks.",
    )
    p.add_argument(
        "--sample-eval-examples",
        action="store_true",
        help="Randomly sample --max-eval-examples instead of taking the first N.",
    )
    p.add_argument("--num-neighbor-hops", type=int, default=None)
    p.add_argument("--fanout", type=int, default=None)
    p.add_argument("--max-nodes-per-type", type=int, default=None)
    return p.parse_args()


def model_from_checkpoint(data, checkpoint: dict[str, object]) -> HGTRecommender:
    config = checkpoint.get("config", {})
    assert isinstance(config, dict)
    model = HGTRecommender.from_data(
        data,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.1)),
        aggregate_layers=str(config.get("aggregate_layers", "mean")),
        normalize=True,
        temperature=float(config.get("temperature", 20.0)),
        use_review_head=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def infer_embeddings(
    model: HGTRecommender,
    subgraph_sampler: HGTSubgraphSampler,
    node_type: str,
    num_nodes: int,
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    embeddings = torch.empty(num_nodes, hidden_dim, dtype=torch.float32)
    all_nodes = torch.arange(num_nodes, dtype=torch.long)

    for start in tqdm(
        range(0, num_nodes, batch_size),
        desc=f"Infer {node_type}",
        dynamic_ncols=True,
    ):
        node_ids = all_nodes[start : start + batch_size]
        batch = subgraph_sampler.sample_nodes(node_type, node_ids)
        out = model.encode(batch.data.to(device))
        key = "h_user_t" if node_type == "user" else "h_item_t"
        emb = out[key][batch.local_node_ids.to(device)].detach().cpu()
        embeddings.index_copy_(0, node_ids, emb)

    return embeddings


@torch.no_grad()
def evaluate_topk(
    model: HGTRecommender,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    eval_users: torch.Tensor,
    eval_items: torch.Tensor,
    user_reviewed_train: list[set[int]],
    ks: tuple[int, ...],
    score_batch_size: int,
    item_chunk_size: int,
    device: torch.device,
) -> dict[str, float]:
    n = eval_users.numel()
    topk_max = min(max(ks), item_emb.size(0))
    if n == 0:
        return {
            **{f"Recall@{k}": 0.0 for k in ks},
            **{f"NDCG@{k}": 0.0 for k in ks},
            **{f"HitRate@{k}": 0.0 for k in ks},
        }

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({f"HitRate@{k}": 0.0 for k in ks})

    item_node_id = torch.arange(item_emb.size(0), dtype=torch.long, device=device)

    for start in tqdm(
        range(0, n, score_batch_size),
        desc="Score",
        dynamic_ncols=True,
    ):
        end = min(start + score_batch_size, n)
        users_cpu = eval_users[start:end].cpu()
        targets = eval_items[start:end].cpu()
        users = users_cpu.to(device)

        batch_user_emb = user_emb.index_select(0, users_cpu).to(device)
        best_scores = None
        best_indices = None

        for item_start in range(0, item_emb.size(0), item_chunk_size):
            item_end = min(item_start + item_chunk_size, item_emb.size(0))
            chunk_items = torch.arange(item_start, item_end, device=device)
            chunk_item_emb = item_emb[item_start:item_end].to(device)

            scores = score_chunk(
                model,
                batch_user_emb,
                chunk_item_emb,
                users,
                chunk_items,
                item_node_id[item_start:item_end],
            )

            for row, (u, pos) in enumerate(zip(users_cpu.tolist(), targets.tolist())):
                seen = user_reviewed_train[u]
                if not seen:
                    continue
                mask = [i - item_start for i in seen if item_start <= i < item_end and i != pos]
                if mask:
                    scores[row, torch.tensor(mask, device=device)] = float("-inf")

            chunk_k = min(topk_max, scores.size(1))
            chunk_scores, chunk_local_idx = torch.topk(scores, k=chunk_k, dim=1)
            chunk_global_idx = chunk_local_idx + item_start

            if best_scores is None:
                best_scores = chunk_scores
                best_indices = chunk_global_idx
            else:
                merged_scores = torch.cat([best_scores, chunk_scores], dim=1)
                merged_indices = torch.cat([best_indices, chunk_global_idx], dim=1)
                best_scores, order = torch.topk(
                    merged_scores, k=topk_max, dim=1
                )
                best_indices = torch.gather(merged_indices, 1, order)

        assert best_indices is not None
        topk = best_indices.cpu()
        matches = topk.eq(targets.unsqueeze(1))

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


def score_chunk(
    model: HGTRecommender,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    item_node_ids: torch.Tensor,
) -> torch.Tensor:
    dot = model.temperature * (user_emb @ item_emb.T)
    score = dot

    if model.rank_mlp is not None:
        num_users = user_emb.size(0)
        num_items = item_emb.size(0)
        pair = torch.cat(
            [
                user_emb[:, None, :].expand(num_users, num_items, -1),
                item_emb[None, :, :].expand(num_users, num_items, -1),
                user_emb[:, None, :] * item_emb[None, :, :],
                torch.abs(user_emb[:, None, :] - item_emb[None, :, :]),
            ],
            dim=-1,
        )
        score = score + model.rank_mlp(pair).squeeze(-1)
    if model.user_bias is not None:
        score = score + model.user_bias(user_ids).squeeze(-1)[:, None]
    if model.item_bias is not None:
        score = score + model.item_bias(item_node_ids).squeeze(-1)[None, :]
    if model.global_bias is not None:
        score = score + model.global_bias
    return score


def main() -> None:
    args = parse_args()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    assert isinstance(config, dict)

    graph_path = args.graph_path or checkpoint.get("graph_path") or config.get("graph_path")
    if graph_path is None:
        raise ValueError("No graph path found. Pass --graph-path.")
    graph_path = Path(graph_path)
    if not graph_path.is_absolute():
        graph_path = PROJECT_ROOT / graph_path

    payload = torch.load(graph_path, weights_only=False)
    data = payload["graph"]
    device = pick_device(args.device)

    model = model_from_checkpoint(data, checkpoint).to(device)
    model.eval()

    rating_threshold = float(
        args.rating_threshold
        if args.rating_threshold is not None
        else config.get("rating_threshold", 4.0)
    )
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    sampler = BPRSampler(data, rating_threshold=rating_threshold, seed=seed)

    subgraph_sampler = HGTSubgraphSampler(
        sampler=sampler,
        num_hops=int(
            args.num_neighbor_hops
            if args.num_neighbor_hops is not None
            else config.get("num_neighbor_hops", 2)
        ),
        fanout=int(args.fanout if args.fanout is not None else config.get("fanout", 8)),
        max_nodes_per_type=int(
            args.max_nodes_per_type
            if args.max_nodes_per_type is not None
            else config.get("max_nodes_per_type", 4096)
        ),
        seed=seed + 97,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Graph: {graph_path}")
    print(f"Device: {device}")
    print(
        "Inference sampling: "
        f"hops={subgraph_sampler.num_hops} fanout={subgraph_sampler.fanout} "
        f"max_nodes_per_type={subgraph_sampler.max_nodes_per_type}"
    )

    hidden_dim = model.encoder.hidden_dim
    user_emb = infer_embeddings(
        model,
        subgraph_sampler,
        "user",
        data["user"].num_nodes,
        hidden_dim,
        args.embedding_batch_size,
        device,
    )
    item_emb = infer_embeddings(
        model,
        subgraph_sampler,
        "item",
        data["item"].num_nodes,
        hidden_dim,
        args.embedding_batch_size,
        device,
    )

    if args.split == "val":
        eval_users = sampler.val_users
        eval_items = sampler.val_items
    else:
        eval_users = sampler.test_users
        eval_items = sampler.test_items

    if args.max_eval_examples is not None and args.max_eval_examples > 0:
        n = min(args.max_eval_examples, eval_users.numel())
        if args.sample_eval_examples:
            gen = torch.Generator().manual_seed(seed + 193)
            idx = torch.randperm(eval_users.numel(), generator=gen)[:n]
        else:
            idx = torch.arange(n)
        eval_users = eval_users.index_select(0, idx)
        eval_items = eval_items.index_select(0, idx)
        print(
            f"Using {n:,} / "
            f"{sampler.val_users.numel() if args.split == 'val' else sampler.test_users.numel():,} "
            f"{args.split} positives for quick evaluation."
        )

    metrics = evaluate_topk(
        model,
        user_emb,
        item_emb,
        eval_users,
        eval_items,
        sampler.user_reviewed_train,
        tuple(args.ks),
        args.score_batch_size,
        args.item_chunk_size,
        device,
    )

    print(f"\n{args.split} metrics:")
    for key, value in metrics.items():
        print(f"  {key:<12s} {value:.4f}")


if __name__ == "__main__":
    main()
