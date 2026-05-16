#!/usr/bin/env python3
"""
Build the MovieLens-100K heterogeneous graph (50K-rating subsample).

Same node-type / edge-type schema as the rest of the pipeline so the
model, sampler, and trainer need no changes:

  ``user`` slot   ← MovieLens user_id
  ``tire`` slot   ← movie
  ``brand`` slot  ← primary genre (first listed genre per movie)
  ``size`` slot   ← release decade ("1990s", "2000s", ...)

Usage
-----
    uv run python scripts/build_graph_movielens.py
    uv run python scripts/build_graph_movielens.py --config configs/movielens.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.graph_builder import (
    create_heterogeneous_graph,
    display_graph_summary,
)
from src.data_processing.preprocessing_movielens import (
    GENRE_COLUMNS,
    load_movielens_data,
    prepare_movielens_dataframes,
    scale_movielens_item_features,
)


def main(config_path: str = "configs/movielens.yaml") -> None:
    with open(PROJECT_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    ratings_path = PROJECT_ROOT / cfg["data"]["ratings_path"]
    users_path = PROJECT_ROOT / cfg["data"]["users_path"]
    items_path = PROJECT_ROOT / cfg["data"]["items_path"]
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    graph_filename = cfg["data"]["graph_filename"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    add_reverse = cfg["graph"]["add_reverse_edges"]
    use_weight = cfg["graph"]["use_rating_as_edge_weight"]

    print(f"Loading MovieLens files ...")
    t0 = time.time()
    ratings, _users, items = load_movielens_data(ratings_path, users_path, items_path)
    print(f"   Loaded {len(ratings):,} ratings in {time.time() - t0:.1f}s")

    print("Building DataFrames ...")
    df_reviews, df_tires, df_brands, df_sizes = prepare_movielens_dataframes(ratings, items)
    print(f"   Reviews:        {len(df_reviews):>6,}")
    print(f"   Users:          {df_reviews['user_id'].nunique():>6,}")
    print(f"   Movies (item):  {len(df_tires):>6,}")
    print(f"   Genres (brand): {len(df_brands):>6,}")
    print(f"   Decades (size): {len(df_sizes):>6,}")

    print("Normalising item features ...")
    item_features, used_cols = scale_movielens_item_features(df_tires)
    print(f"   Feature matrix shape: {item_features.shape}")
    print(f"   Columns: {used_cols}")

    print("Building heterogeneous graph ...")
    data, mappings = create_heterogeneous_graph(
        df_reviews=df_reviews,
        df_tires=df_tires,
        df_brands=df_brands,
        df_sizes=df_sizes,
        tire_features=item_features,
        add_reverse_edges=add_reverse,
        use_rating_as_edge_weight=use_weight,
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
    assert not torch.isnan(data["tire"].x).any(), "NaN in item features"
    print("   All checks passed!")

    fw_edge = ("user", "reviews", "tire")
    n_edges = data[fw_edge].edge_index.size(1)
    print(f"\nDensity check:")
    print(f"   reviews/user   : {n_edges / data['user'].num_nodes:.2f}")
    print(f"   reviews/movie  : {n_edges / data['tire'].num_nodes:.2f}")

    # Attach the static (year + 19 genre flags) block to review_df.attrs so
    # the sampler can recompute aggregates from train rows only without losing
    # the static columns. The static block is the last 20 columns of the
    # normalised feature matrix (continuous 4 cols → first 4 are scaled aggregates).
    # Layout from scale_movielens_item_features:
    #   [avg_rating, rating_std, rating_number, release_year, *19 genres]
    # → static = columns 3..23 (release_year + 19 genres).
    df_reviews.attrs["static_feats"] = item_features[:, 3:].copy()

    out_path = processed_dir / graph_filename
    payload = {
        "graph": data,
        "mappings": mappings,
        "tire_df": df_tires,
        "review_df": df_reviews,
    }
    torch.save(payload, out_path)
    print(f"\nGraph saved to {out_path}")

    loaded = torch.load(out_path, weights_only=False)
    assert loaded["graph"].node_types == data.node_types
    print(
        "   Mappings: "
        + ", ".join(f"{k}: {len(v)}" for k, v in mappings.items())
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MovieLens-100K hetero graph")
    parser.add_argument("--config", default="configs/movielens.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
