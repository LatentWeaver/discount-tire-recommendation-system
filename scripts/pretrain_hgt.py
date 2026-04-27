#!/usr/bin/env python3
"""
Pretrain HGT item representations on a review graph.

This script is meant to run before user-feedback fine-tuning. It learns:
  - user-item ranking signal from BPR
  - liked vs. disliked signal from observed review ratings
  - metadata structure through HGT message passing

Usage
-----
    uv run python scripts/pretrain_hgt.py
    uv run python scripts/pretrain_hgt.py --epochs 5 --steps-per-epoch 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hgt_recommender import HGTRecommender
from src.training.hgt_pretrainer import HGTPretrainer
from src.training.pyg_link_sampler import (
    PyGLinkNeighborBatcher,
    has_compiled_neighbor_sampler,
)
from src.training.sampler import BPRSampler
from src.training.subgraph_sampler import HGTSubgraphSampler


def pick_device(preferred: str | None) -> torch.device:
    """cuda > mps > cpu, with optional manual override."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain HGT item recommender.")
    p.add_argument(
        "--graph-path",
        type=str,
        default="data/processed/lastfm_hetero_graph.pt",
        help="Graph payload produced by scripts/build_lastfm_graph.py.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--training-mode",
        choices=("subgraph", "full"),
        default="subgraph",
        help="Use HGT-style mini-batch subgraph sampling or full-graph training.",
    )
    p.add_argument(
        "--sampler-backend",
        choices=("auto", "pyg-link", "python"),
        default="auto",
        help="Subgraph sampler backend. auto uses PyG compiled link sampling on CUDA when available.",
    )
    p.add_argument("--num-neighbor-hops", type=int, default=2)
    p.add_argument(
        "--fanout",
        type=int,
        default=8,
        help="Maximum sampled incident neighbors per node, relation, and hop. <=0 keeps all.",
    )
    p.add_argument(
        "--max-nodes-per-type",
        type=int,
        default=4096,
        help="Maximum nodes retained per node type in sampled subgraphs. <=0 disables the cap.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--aggregate-layers",
        choices=("mean", "last"),
        default="mean",
        help="Use LightGCN-style mean pooling over input/layer embeddings or last layer only.",
    )
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--rating-threshold", type=float, default=4.0)
    p.add_argument(
        "--review-loss-weight",
        type=float,
        default=0.5,
        help="Weight for observed liked/disliked review classification loss.",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run full-graph validation every N epochs. 0 disables full-graph eval.",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default="outputs/checkpoints/hgt_pretrained.pt",
        help="Where to save the best pretrain checkpoint.",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    graph_path = (PROJECT_ROOT / args.graph_path).resolve()
    payload = torch.load(graph_path, weights_only=False)
    data = payload["graph"]

    device = pick_device(args.device)
    train_data_for_model = data.to(device) if args.training_mode == "full" else data

    model = HGTRecommender.from_data(
        train_data_for_model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        aggregate_layers=args.aggregate_layers,
        normalize=True,
        temperature=args.temperature,
        use_review_head=True,
    ).to(device)

    sampler = BPRSampler(
        train_data_for_model,
        rating_threshold=args.rating_threshold,
        seed=args.seed,
    )
    use_pyg_link = (
        args.training_mode == "subgraph"
        and args.sampler_backend in {"auto", "pyg-link"}
        and device.type == "cuda"
        and has_compiled_neighbor_sampler()
    )
    if args.sampler_backend == "pyg-link" and not use_pyg_link:
        raise RuntimeError(
            "--sampler-backend pyg-link requires CUDA plus pyg-lib or torch-sparse. "
            "Install pyg-lib on the remote CUDA machine, or use --sampler-backend python."
        )

    pyg_link_batcher = (
        PyGLinkNeighborBatcher(
            sampler=sampler,
            batch_size=args.batch_size,
            num_hops=args.num_neighbor_hops,
            fanout=args.fanout,
            neg_sampling_ratio=args.neg_sampling_ratio,
            num_workers=args.num_workers,
        )
        if use_pyg_link
        else None
    )
    subgraph_sampler = (
        HGTSubgraphSampler(
            sampler=sampler,
            num_hops=args.num_neighbor_hops,
            fanout=args.fanout,
            max_nodes_per_type=args.max_nodes_per_type,
            seed=args.seed + 7,
        )
        if args.training_mode == "subgraph" and pyg_link_batcher is None
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    trainer = HGTPretrainer(
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        review_loss_weight=args.review_loss_weight,
        subgraph_sampler=subgraph_sampler,
        pyg_link_batcher=pyg_link_batcher,
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Graph: {graph_path}")
    print(
        f"Device: {device} | Params: {n_params:,} | "
        f"Training: {args.training_mode}"
    )
    if args.training_mode == "subgraph":
        print(
            f"Subgraph sampling: hops={args.num_neighbor_hops} "
            f"fanout={args.fanout} max_nodes_per_type={args.max_nodes_per_type} "
            f"backend={'pyg-link' if pyg_link_batcher is not None else 'python'}"
        )
        if pyg_link_batcher is None and device.type == "cuda":
            print(
                "Warning: using Python sampler on CUDA. Install pyg-lib for faster "
                "compiled sampling, then run with --sampler-backend pyg-link."
            )
    print(
        f"Train positives: {sampler.train_users.size(0):,} | "
        f"Observed train reviews: {sampler.train_observed_users.size(0):,} | "
        f"Val positives: {sampler.val_users.size(0):,} | "
        f"Test positives: {sampler.test_users.size(0):,}"
    )
    print(
        f"Review labels: train liked ratio = "
        f"{sampler.train_observed_labels.mean().item():.1%}"
    )

    best_val = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        agg = {"loss": 0.0, "L_bpr": 0.0, "L_review": 0.0}
        epoch_start = time.time()
        progress = tqdm(
            range(args.steps_per_epoch),
            desc=f"Pretrain {epoch:>3d}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in progress:
            stats = trainer.train_step(batch_size=args.batch_size)
            for k in agg:
                agg[k] += stats[k]
            progress.set_postfix(loss=f"{stats['loss']:.4f}")
        for k in agg:
            agg[k] /= args.steps_per_epoch

        msg = (
            f"[epoch {epoch:>3d}] "
            f"time={time.time() - epoch_start:.1f}s "
            f"loss={agg['loss']:.4f} "
            f"BPR={agg['L_bpr']:.4f} "
            f"review={agg['L_review']:.4f}"
        )

        if args.eval_every > 0 and (
            epoch % args.eval_every == 0 or epoch == args.epochs
        ):
            metrics = trainer.evaluate(split="val", ks=(10, 20, 50))
            review_metrics = trainer.review_metrics(split="val")
            msg += (
                f"  Recall@20={metrics['Recall@20']:.4f} "
                f"ReviewBalAcc={review_metrics['ReviewBalancedAcc']:.4f}"
            )
            val_score = metrics["Recall@20"]
            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                msg += "  * best"

        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best pretrain checkpoint from epoch {best_epoch}")

    if args.eval_every > 0:
        print("Final test metrics:")
        test_metrics = trainer.evaluate(split="test", ks=(10, 20, 50))
        test_review_metrics = trainer.review_metrics(split="test")
        for k, v in test_metrics.items():
            print(f"  {k:<14s} {v:.4f}")
        for k, v in test_review_metrics.items():
            print(f"  {k:<18s} {v:.4f}")
    else:
        print("\nSkipped full-graph validation/test metrics (--eval-every 0).")

    save_path = (PROJECT_ROOT / args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "graph_path": args.graph_path,
            "graph_path_resolved": str(graph_path),
            "config": vars(args),
            "best_epoch": best_epoch,
            "best_val_recall_at_20": best_val,
        },
        save_path,
    )
    print(f"\nSaved pretrained checkpoint -> {save_path}")


if __name__ == "__main__":
    main()
