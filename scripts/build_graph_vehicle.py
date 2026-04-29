#!/usr/bin/env python3
"""
Build the vehicle-as-query heterogeneous graph from the
Discount-Tire-RGCN per-product JSON folder.

The graph keeps the *exact same* node-type / edge-type schema as
``build_graph.py`` so the model, sampler, and trainer need no changes:

  ``user`` slot   ← vehicle (make, model) pairs
  ``tire`` slot   ← tire products
  ``brand`` slot  ← parsed first-token of product_name
  ``size`` slot   ← single placeholder ("Mixed")  — this dataset has no per-review tire size

Usage
-----
    uv run python scripts/build_graph_vehicle.py
    uv run python scripts/build_graph_vehicle.py --config configs/vehicle.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.graph_builder import (
    create_heterogeneous_graph,
    display_graph_summary,
)
from src.data_processing.preprocessing_vehicle import (
    load_vehicle_review_data,
    prepare_vehicle_dataframes,
    scale_vehicle_tire_features,
)


def main(config_path: str = "configs/vehicle.yaml") -> None:
    with open(PROJECT_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    graph_filename = cfg["data"]["graph_filename"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    add_reverse = cfg["graph"]["add_reverse_edges"]
    use_weight = cfg["graph"]["use_rating_as_edge_weight"]

    print(f"Loading per-product reviews from {raw_dir} ...")
    t0 = time.time()
    records = load_vehicle_review_data(raw_dir)
    print(f"   Loaded {len(records):,} reviews in {time.time() - t0:.1f}s")

    print("Building DataFrames ...")
    df_reviews, df_tires, df_brands, df_sizes = prepare_vehicle_dataframes(records)
    print(f"   Reviews:        {len(df_reviews):>6,}")
    print(f"   Vehicles (user):{df_reviews['user_id'].nunique():>6,}  "
          f"(distinct (make,model))")
    print(f"   Tires:          {len(df_tires):>6,}")
    print(f"   Brands:         {len(df_brands):>6,}")
    print(f"   Sizes:          {len(df_sizes):>6,}  (placeholder)")

    print("Normalising tire features ...")
    tire_features, used_cols = scale_vehicle_tire_features(df_tires)
    print(f"   Feature matrix shape: {tire_features.shape}")
    print(f"   Columns: {used_cols}")

    print("Building heterogeneous graph ...")
    data, mappings = create_heterogeneous_graph(
        df_reviews=df_reviews,
        df_tires=df_tires,
        df_brands=df_brands,
        df_sizes=df_sizes,
        tire_features=tire_features,
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
    assert not torch.isnan(data["tire"].x).any(), "NaN in tire features"
    print("   All checks passed!")

    # Density sanity — this is the whole point of the rebuild.
    fw_edge = ("user", "reviews", "tire")
    n_edges = data[fw_edge].edge_index.size(1)
    print(f"\nDensity check:")
    print(f"   reviews/vehicle  : {n_edges / data['user'].num_nodes:.2f}")
    print(f"   reviews/tire     : {n_edges / data['tire'].num_nodes:.2f}")

    # Optional: attach precomputed sentence-transformer review-text embeddings.
    text_emb_path = processed_dir / "tire_text_emb_vehicle.npy"
    if text_emb_path.exists():
        import numpy as np
        text_emb = np.load(text_emb_path).astype(np.float32)
        if text_emb.shape[0] != data["tire"].num_nodes:
            raise ValueError(
                f"Text embedding row count ({text_emb.shape[0]}) does not match"
                f" tire node count ({data['tire'].num_nodes}). Re-run"
                f" scripts/build_review_text_embeddings.py."
            )
        data["tire"].text_x = torch.from_numpy(text_emb)
        print(f"Attached review-text embeddings: {tuple(data['tire'].text_x.shape)}")
    else:
        print(
            "No review-text embeddings found at "
            f"{text_emb_path.relative_to(PROJECT_ROOT)}. "
            "Run scripts/build_review_text_embeddings.py to enable text fusion."
        )

    out_path = processed_dir / graph_filename
    payload = {
        "graph": data,
        "mappings": mappings,
        "tire_df": df_tires,
        "review_df": df_reviews,  # keeps sub_ratings for future multi-task heads
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
    parser = argparse.ArgumentParser(description="Build vehicle-as-query tire graph")
    parser.add_argument("--config", default="configs/vehicle.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
