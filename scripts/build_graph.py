#!/usr/bin/env python3
"""
Build the heterogeneous graph from raw JSONL data.

Usage
-----
    uv run python scripts/build_graph.py
    uv run python scripts/build_graph.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.preprocessing import (
    prepare_dataframes,
    load_review_data,
    scale_tire_features,
)
from src.data_processing.graph_builder import create_heterogeneous_graph, display_graph_summary


def main(config_path: str = "configs/default.yaml") -> None:
    # ── Load config ───────────────────────────────────────
    with open(PROJECT_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]
    if not raw_path.exists():
        fallback = PROJECT_ROOT / Path(cfg["data"]["raw_path"]).name
        if fallback.exists():
            print(f"Configured raw file not found, using fallback: {fallback}")
            raw_path = fallback
        else:
            raise FileNotFoundError(
                f"Raw dataset not found at {raw_path} or {fallback}"
            )
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    graph_filename = cfg["data"]["graph_filename"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    add_reverse = cfg["graph"]["add_reverse_edges"]
    use_weight = cfg["graph"]["use_rating_as_edge_weight"]

    # ── Load & parse JSONL ────────────────────────
    print(f"Loading data from {raw_path} ...")
    t0 = time.time()
    records = load_review_data(raw_path)
    print(f"   Loaded {len(records):,} records in {time.time() - t0:.1f}s")

    # ── Build DataFrames ──────────────────────────
    print("Building DataFrames ...")
    df_reviews, df_tires, df_brands, df_sizes = prepare_dataframes(records)
    print(f"   Reviews:  {len(df_reviews):>6,}")
    print(f"   Tires:    {len(df_tires):>6,}")
    print(f"   Brands:   {len(df_brands):>6,}")
    print(f"   Sizes:    {len(df_sizes):>6,}")

    # ── Normalise tire features ───────────────────
    print("Normalising tire features ...")
    cont_cols = cfg["tire_features"]["continuous"]
    cat_cols = cfg["tire_features"]["categorical"]
    tire_features, encoders = scale_tire_features(
        df_tires, continuous_cols=cont_cols, categorical_cols=cat_cols
    )
    print(f"   Feature matrix shape: {tire_features.shape}")

    # ── Build graph ───────────────────────────────
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

    # ── Sanity checks ─────────────────────────────
    print("\nRunning sanity checks ...")
    for ntype in data.node_types:
        assert data[ntype].num_nodes > 0, f"No nodes for type {ntype}"
    for etype in data.edge_types:
        ei = data[etype].edge_index
        src_type, _, dst_type = etype
        assert ei.shape[0] == 2
        assert ei.min() >= 0
        assert ei[0].max() < data[src_type].num_nodes, (
            f"Edge src out of range for {etype}"
        )
        assert ei[1].max() < data[dst_type].num_nodes, (
            f"Edge dst out of range for {etype}"
        )
    # Check for NaN in tire features
    assert not torch.isnan(data["tire"].x).any(), "NaN found in tire features"
    print("   All checks passed!")

    # ── Save ──────────────────────────────────────
    # Bundle graph + ID mappings + tire metadata so that downstream scripts
    # (train, eval, visualize) can resolve human-readable names.
    out_path = processed_dir / graph_filename
    payload = {
        "graph": data,
        "mappings": mappings,       # {user_map, tire_map, brand_map, size_map}
        "tire_df": df_tires,        # original tire DataFrame (has price, brand, size, etc.)
    }
    torch.save(payload, out_path)
    print(f"\nGraph + mappings saved to {out_path}")

    # Quick reload test
    loaded = torch.load(out_path, weights_only=False)
    assert loaded["graph"].node_types == data.node_types
    assert "mappings" in loaded
    assert "tire_df" in loaded
    print("   Reload verification passed!")
    print(f"   Mappings: {', '.join(f'{k}: {len(v)} entries' for k, v in mappings.items())}")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build heterogeneous tire graph")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()
    main(config_path=args.config)
