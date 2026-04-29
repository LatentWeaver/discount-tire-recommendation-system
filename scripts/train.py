#!/usr/bin/env python3
"""
End-to-end Two-Tower training script.

  HGT encoder → ItemTower / UserTower
  Loss (default): sampled-softmax over in-batch negatives
  Eval: Recall@K / NDCG@K / HitRate@K on the val split.

Usage
-----
    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 30 --batch-size 512 --lr 5e-4
    uv run python scripts/train.py --loss bpr
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# MPS fallback for any PyG scatter op not yet implemented.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pick_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enable_cuda_perf_flags() -> None:
    """TF32 matmul + cuDNN benchmark — free speedup for fixed-shape forwards."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


from src.models import TwoTowerRecommender
from src.training.sampler import BPRSampler
from src.training.trainer import TwoTowerTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Two-Tower tire recommender.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--out-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--init-temperature", type=float, default=0.07)
    p.add_argument("--loss", type=str, default="softmax",
                   choices=["softmax", "bpr", "bce"])
    p.add_argument("--amp", action="store_true",
                   help="Enable bf16 autocast on CUDA (≈1.5-2× speedup on Ampere+).")
    # SGL-style self-supervised contrastive.
    p.add_argument("--ssl-lambda", type=float, default=0.0,
                   help="Weight on the SGL contrastive loss. 0 disables. Try 0.1.")
    p.add_argument("--ssl-edge-drop", type=float, default=0.2)
    p.add_argument("--ssl-feat-drop", type=float, default=0.1)
    p.add_argument("--ssl-tau", type=float, default=0.5)
    p.add_argument("--ssl-sample-size", type=int, default=1024,
                   help="Subset of users + tires used per InfoNCE step.")
    # History-pool dropout.
    p.add_argument("--history-drop", type=float, default=0.0,
                   help="Probability of dropping each (user, train-positive) pair "
                        "before mean-pooling. 0 disables. Try 0.3.")
    # Hard-negative mining.
    p.add_argument("--hard-neg-k", type=int, default=0,
                   help="Hard negatives per positive, drawn from (brand ∪ size) buckets. "
                        "0 disables. Try 4–8 for softmax; 1 for bpr/bce.")
    p.add_argument("--rating-threshold", type=float, default=4.0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--best-metric", type=str, default="Recall@20")
    p.add_argument("--save-path", type=str,
                   default="outputs/checkpoints/two_tower.pt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph-path", type=str,
                   default="data/processed/hetero_graph.pt",
                   help="Relative-to-project-root path of the .pt graph payload.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    graph_path = PROJECT_ROOT / args.graph_path
    payload = torch.load(graph_path, weights_only=False)
    print(f"Loaded graph payload from {graph_path}")
    data = payload["graph"]

    device = pick_device(args.device)
    if device.type == "cuda":
        enable_cuda_perf_flags()
    data = data.to(device)

    sampler = BPRSampler(
        data,
        rating_threshold=args.rating_threshold,
        seed=args.seed,
        review_df=payload.get("review_df"),
    )

    model = TwoTowerRecommender.from_data(
        sampler.train_data,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        out_dim=args.out_dim,
        dropout=args.dropout,
        init_temperature=args.init_temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    trainer = TwoTowerTrainer(
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        loss=args.loss,
        amp=args.amp,
        ssl_lambda=args.ssl_lambda,
        ssl_edge_drop=args.ssl_edge_drop,
        ssl_feat_drop=args.ssl_feat_drop,
        ssl_tau=args.ssl_tau,
        ssl_sample_size=args.ssl_sample_size,
        history_drop=args.history_drop,
        hard_neg_k=args.hard_neg_k,
        seed=args.seed,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    amp_tag = " amp(bf16)" if trainer.amp else ""
    text_tag = " +text" if model.uses_text else ""
    ssl_tag = f" +ssl(λ={args.ssl_lambda})" if args.ssl_lambda > 0 else ""
    hdrop_tag = f" +hdrop({args.history_drop})" if args.history_drop > 0 else ""
    hneg_tag = f" +hneg(k={args.hard_neg_k})" if args.hard_neg_k > 0 else ""
    print(
        f"Device: {device}{amp_tag} | Params: {n_params:,} | "
        f"Loss: {args.loss}{text_tag}{ssl_tag}{hdrop_tag}{hneg_tag}"
    )
    print(
        f"Train: {sampler.train_users.size(0):,} | "
        f"Val: {sampler.val_users.size(0):,} | "
        f"Test: {sampler.test_users.size(0):,}"
    )

    best_val = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        agg = {"loss": 0.0, "L_main": 0.0, "L_ssl": 0.0}
        epoch_start = time.time()
        progress = tqdm(
            range(args.steps_per_epoch),
            desc=f"Epoch {epoch:>3d}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in progress:
            stats = trainer.train_step(batch_size=args.batch_size)
            for k in agg:
                agg[k] += stats[k]
            progress.set_postfix(
                loss=f"{stats['loss']:.4f}",
                main=f"{stats['L_main']:.4f}",
                ssl=f"{stats['L_ssl']:.4f}",
                temp=f"{stats['temp']:.3f}",
            )
        for k in agg:
            agg[k] /= args.steps_per_epoch
        epoch_time = time.time() - epoch_start

        msg = (
            f"[epoch {epoch:>3d}] time={epoch_time:.1f}s "
            f"loss={agg['loss']:.4f} "
            f"main={agg['L_main']:.4f} "
            f"ssl={agg['L_ssl']:.4f} "
            f"temp={float(model.temperature.item()):.3f}"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = trainer.evaluate(split="val", ks=(10, 20, 50))
            metric_str = " | ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("Recall")
            )
            msg += f"  {metric_str}"

            val_score = metrics.get(args.best_metric)
            if val_score is None:
                raise KeyError(
                    f"--best-metric={args.best_metric!r} not in eval output "
                    f"(keys: {sorted(metrics)})"
                )
            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                msg += f"  * best {args.best_metric}"

        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"\nRestored best checkpoint from epoch {best_epoch} "
            f"({args.best_metric}={best_val:.4f})"
        )

    print("\nFinal test metrics:")
    test_metrics = trainer.evaluate(split="test", ks=(10, 20, 50))
    for k, v in test_metrics.items():
        print(f"  {k:<14s} {v:.4f}")

    p = Path(args.save_path)
    save_path = (PROJECT_ROOT / p.parent / f"{p.stem}_e{args.epochs}{p.suffix}").resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "split_seed": args.seed,
    }, save_path)
    print(f"\nSaved best weights → {save_path}")


if __name__ == "__main__":
    main()
