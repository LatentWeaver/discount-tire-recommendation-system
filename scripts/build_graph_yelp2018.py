#!/usr/bin/env python3
"""
Build the LightGCN Yelp 2018 heterogeneous graph.

Source: https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018

Same node-type / edge-type schema as the rest of the pipeline so the
model, sampler, and trainer need no changes:

  ``user`` slot   ← LightGCN user_id   (31,668 nodes)
  ``tire`` slot   ← LightGCN business  (38,048 nodes)
  ``brand`` slot  ← single "ALL" node  (LightGCN has no item categories)
  ``size`` slot   ← single "ALL" node  (LightGCN has no item attributes)

LightGCN's train / test split is honored verbatim — every train.txt and
test.txt interaction is concatenated into a single ``(user, reviews, tire)``
edge index in that order, and a ``precomputed_split`` dict on the saved
payload records the exact index ranges so the sampler can use them
instead of re-splitting. A small validation slice is carved out of the
train edges for early-stopping; LightGCN's test split is never touched.

Usage
-----
    uv run python scripts/build_graph_yelp2018.py
    uv run python scripts/build_graph_yelp2018.py --config configs/yelp2018.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.graph_builder import display_graph_summary
from src.data_processing.preprocessing_yelp2018 import (
    carve_val_from_train,
    load_lightgcn_yelp2018,
)


def build_yelp2018_graph(
    num_users: int,
    num_items: int,
    train_edges: np.ndarray,
    val_edges: np.ndarray,
    test_edges: np.ndarray,
    add_reverse_edges: bool = True,
) -> tuple[HeteroData, dict[str, torch.Tensor]]:
    """Materialise a HeteroData graph from LightGCN edge arrays.

    The user→tire edge index is the concatenation of [train, val, test]
    in that order; the returned ``precomputed_split`` records the row
    ranges so the sampler can recover each partition without re-splitting.
    """
    n_train = train_edges.shape[0]
    n_val = val_edges.shape[0]
    n_test = test_edges.shape[0]

    parts = [train_edges]
    if n_val > 0:
        parts.append(val_edges)
    parts.append(test_edges)
    edges = np.concatenate(parts, axis=0)        # (N, 2)
    edge_index = torch.from_numpy(edges.T).contiguous().long()  # (2, N)

    data = HeteroData()

    data["user"].num_nodes = num_users
    data["user"].node_id = torch.arange(num_users)

    # No item content features in this release. Carry a zero feature column
    # so spec_dim is well-defined and the ItemTower's tire_specs channel
    # stays a no-op gradient path (every item gets the same constant 0).
    data["tire"].x = torch.zeros((num_items, 1), dtype=torch.float32)
    data["tire"].num_nodes = num_items

    # Single dummy brand and size nodes so the model's HGT slots remain
    # populated. Every item routes through the same brand/size embedding.
    data["brand"].num_nodes = 1
    data["brand"].node_id = torch.arange(1)
    data["size"].num_nodes = 1
    data["size"].node_id = torch.arange(1)

    data["user", "reviews", "tire"].edge_index = edge_index
    if add_reverse_edges:
        data["tire", "rev_by", "user"].edge_index = edge_index.flip(0)

    item_to_brand = torch.stack(
        [torch.arange(num_items), torch.zeros(num_items, dtype=torch.long)]
    )
    data["tire", "belongs_to", "brand"].edge_index = item_to_brand
    if add_reverse_edges:
        data["brand", "has", "tire"].edge_index = item_to_brand.flip(0)

    item_to_size = item_to_brand.clone()
    data["tire", "has_spec", "size"].edge_index = item_to_size
    if add_reverse_edges:
        data["size", "spec_of", "tire"].edge_index = item_to_size.flip(0)

    train_idx = torch.arange(0, n_train, dtype=torch.long)
    val_idx = torch.arange(n_train, n_train + n_val, dtype=torch.long)
    test_idx = torch.arange(n_train + n_val, n_train + n_val + n_test, dtype=torch.long)
    precomputed_split = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    return data, precomputed_split


def _subsample_edges(edges: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Randomly keep ``ratio`` fraction of rows from ``edges``."""
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(edges.shape[0] * ratio)))
    idx = rng.choice(edges.shape[0], size=n_keep, replace=False)
    idx.sort()
    return edges[idx]


def main(
    config_path: str = "configs/yelp2018.yaml",
    subsample_ratio: float | None = None,
) -> None:
    with open(PROJECT_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    graph_filename = cfg["data"]["graph_filename"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    add_reverse = cfg["graph"]["add_reverse_edges"]
    val_ratio = float(cfg["graph"]["val_ratio"])
    seed = int(cfg["graph"]["seed"])

    print(f"Loading LightGCN Yelp 2018 files from {raw_dir} ...")
    t0 = time.time()
    interactions = load_lightgcn_yelp2018(raw_dir)
    print(
        f"   {interactions.num_users:,} users · "
        f"{interactions.num_items:,} items · "
        f"{interactions.train_edges.shape[0]:,} train · "
        f"{interactions.test_edges.shape[0]:,} test  ({time.time() - t0:.1f}s)"
    )

    if subsample_ratio is not None:
        if not 0.0 < subsample_ratio < 1.0:
            raise ValueError(
                f"--subsample-ratio must be in (0, 1); got {subsample_ratio}."
            )
        print(f"Subsampling {subsample_ratio:.0%} of train and test edges (seed={seed}) ...")
        sub_train = _subsample_edges(interactions.train_edges, subsample_ratio, seed)
        sub_test = _subsample_edges(interactions.test_edges, subsample_ratio, seed + 1)
        interactions = type(interactions)(
            num_users=interactions.num_users,
            num_items=interactions.num_items,
            train_edges=sub_train,
            test_edges=sub_test,
        )
        # Tag the output filename so the full graph isn't overwritten.
        pct = int(round(subsample_ratio * 100))
        stem, _, ext = graph_filename.rpartition(".")
        graph_filename = f"{stem}_sub{pct:02d}.{ext}"
        print(
            f"   After subsample: "
            f"{sub_train.shape[0]:,} train · {sub_test.shape[0]:,} test → "
            f"{graph_filename}"
        )

    print(f"Carving {val_ratio:.0%} of train into a validation slice (seed={seed}) ...")
    train_edges, val_edges = carve_val_from_train(
        interactions.train_edges, val_ratio=val_ratio, seed=seed
    )
    print(
        f"   Train (after carve): {train_edges.shape[0]:,}  ·  "
        f"Val: {val_edges.shape[0]:,}  ·  "
        f"Test (LightGCN verbatim): {interactions.test_edges.shape[0]:,}"
    )

    print("Building heterogeneous graph ...")
    data, precomputed_split = build_yelp2018_graph(
        num_users=interactions.num_users,
        num_items=interactions.num_items,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=interactions.test_edges,
        add_reverse_edges=add_reverse,
    )
    display_graph_summary(data)

    print("\nRunning sanity checks ...")
    for ntype in data.node_types:
        assert data[ntype].num_nodes > 0, f"No nodes for type {ntype}"
    for etype in data.edge_types:
        ei = data[etype].edge_index
        src_type, _, dst_type = etype
        assert ei.shape[0] == 2
        assert ei.min() >= 0
        assert ei[0].max() < data[src_type].num_nodes
        assert ei[1].max() < data[dst_type].num_nodes
    fw_edge = ("user", "reviews", "tire")
    n_edges = data[fw_edge].edge_index.size(1)
    assert n_edges == (
        precomputed_split["train_idx"].numel()
        + precomputed_split["val_idx"].numel()
        + precomputed_split["test_idx"].numel()
    )
    print("   All checks passed!")

    print(f"\nDensity:")
    print(f"   reviews/user : {n_edges / data['user'].num_nodes:.2f}")
    print(f"   reviews/item : {n_edges / data['tire'].num_nodes:.2f}")

    out_path = processed_dir / graph_filename
    payload = {
        "graph": data,
        "precomputed_split": precomputed_split,
        "num_users": interactions.num_users,
        "num_items": interactions.num_items,
    }
    torch.save(payload, out_path)
    print(f"\nGraph saved to {out_path}")

    loaded = torch.load(out_path, weights_only=False)
    assert loaded["graph"].node_types == data.node_types
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LightGCN Yelp 2018 hetero graph")
    parser.add_argument("--config", default="configs/yelp2018.yaml")
    parser.add_argument(
        "--subsample-ratio",
        type=float,
        default=None,
        help="Randomly keep this fraction of train AND test edges (e.g., 0.1). "
             "Used for fast smoke tests; output filename gets a _subXX suffix.",
    )
    args = parser.parse_args()
    main(config_path=args.config, subsample_ratio=args.subsample_ratio)
