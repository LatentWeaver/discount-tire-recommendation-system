#!/usr/bin/env python3
"""
End-to-end training script.

  HGT encoder → Intermediate (Path A + B) → FusionMLP
  Loss: L_BPR + λ · L_cluster
  Pseudo-labels refreshed every ``--refresh-every`` epochs.
  Eval: Recall@K / NDCG@K / HitRate@K on the val split.

Usage
-----
    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 30 --batch-size 2048 --lr 5e-4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow CPU fallback for any MPS op not yet implemented (some PyG scatter
# variants fall into this bucket). Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pick_device(preferred: str | None) -> torch.device:
    """cuda > mps > cpu, with optional manual override."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

from src.models import TireRecommender
from src.training.sampler import BPRSampler
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the tire recommender.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-clusters", type=int, default=50)
    p.add_argument("--cluster-lambda", type=float, default=0.5)
    p.add_argument("--contrast-lambda", type=float, default=0.3,
                   help="Weight on the bad-vs-good contrastive loss. 0 disables it.")
    p.add_argument("--contrast-batch-size", type=int, default=None,
                   help="Defaults to --batch-size if unset.")
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--refresh-every", type=int, default=2,
                   help="Re-run k-means every N epochs.")
    p.add_argument("--rating-threshold", type=float, default=4.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-path", type=str,
                   default="outputs/checkpoints/recommender.pt",
                   help="Where to save the final model state_dict.")
    p.add_argument("--device", type=str, default=None,
                   help="Force device: cuda / mps / cpu. Defaults to best available.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    graph_path = PROJECT_ROOT / "data" / "processed" / "hetero_graph.pt"
    data = torch.load(graph_path, weights_only=False)["graph"]

    device = pick_device(args.device)
    data = data.to(device)

    model = TireRecommender.from_data(
        data,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_clusters=args.num_clusters,
    ).to(device)

    sampler = BPRSampler(
        data, rating_threshold=args.rating_threshold, seed=args.seed
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    trainer = Trainer(
        model=model,
        data=data,
        sampler=sampler,
        optimizer=optimizer,
        cluster_lambda=args.cluster_lambda,
        contrast_lambda=args.contrast_lambda,
        contrast_batch_size=args.contrast_batch_size,
        pca_dim=args.pca_dim,
        num_clusters=args.num_clusters,
        seed=args.seed,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | Params: {n_params:,}")
    print(
        f"Train: {sampler.train_users.size(0):,} | "
        f"Val: {sampler.val_users.size(0):,} | "
        f"Test: {sampler.test_users.size(0):,}"
    )

    for epoch in range(1, args.epochs + 1):
        if (epoch - 1) % args.refresh_every == 0:
            trainer.refresh_pseudo_labels()
            print(f"[epoch {epoch:>3d}] refreshed pseudo-labels")

        agg = {"loss": 0.0, "L_bpr": 0.0, "L_cluster": 0.0, "L_contrast": 0.0}
        for _ in range(args.steps_per_epoch):
            stats = trainer.train_step(batch_size=args.batch_size)
            for k in agg:
                agg[k] += stats[k]
        for k in agg:
            agg[k] /= args.steps_per_epoch

        msg = (
            f"[epoch {epoch:>3d}] "
            f"loss={agg['loss']:.4f} "
            f"BPR={agg['L_bpr']:.4f} "
            f"cluster={agg['L_cluster']:.4f} "
            f"contrast={agg['L_contrast']:.4f}"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = trainer.evaluate(split="val", ks=(10, 20, 50))
            metric_str = " | ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("Recall")
            )
            msg += f"  {metric_str}"

        print(msg)

    print("\nFinal test metrics:")
    test_metrics = trainer.evaluate(split="test", ks=(10, 20, 50))
    for k, v in test_metrics.items():
        print(f"  {k:<14s} {v:.4f}")

    p = Path(args.save_path)
    save_path = (PROJECT_ROOT / p.parent / f"{p.stem}_e{args.epochs}{p.suffix}").resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved final weights → {save_path}")


if __name__ == "__main__":
    main()
