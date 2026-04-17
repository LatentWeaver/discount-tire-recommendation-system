"""
Heterogeneous graph builder for the tire recommendation system.

Converts preprocessed DataFrames into a PyTorch Geometric ``HeteroData``
object following the schema described in the README:

Node types : user, tire, brand, size
Edge types : (user, reviews, tire),   (tire, rev_by, user)
             (tire, belongs_to, brand), (brand, has, tire)
             (tire, has_spec, size),   (size, spec_of, tire)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def create_index_mapping(values: pd.Series | np.ndarray) -> dict[str, int]:
    """Create a contiguous int mapping from unique string values."""
    unique = pd.Series(values).unique()
    return {v: i for i, v in enumerate(unique)}


def create_heterogeneous_graph(
    df_reviews: pd.DataFrame,
    df_tires: pd.DataFrame,
    df_brands: pd.DataFrame,
    df_sizes: pd.DataFrame,
    tire_features: np.ndarray,
    add_reverse_edges: bool = True,
    use_rating_as_edge_weight: bool = True,
) -> HeteroData:
    """
    Construct a ``HeteroData`` graph from the preprocessed DataFrames.

    Parameters
    ----------
    df_reviews : DataFrame with columns [user_id, tire_title, rating]
    df_tires   : DataFrame with columns [tire_title, brand, size, ...]
    df_brands  : DataFrame with column  [brand]
    df_sizes   : DataFrame with column  [size]
    tire_features : ndarray (num_tires, num_features) — normalised feature matrix
    add_reverse_edges : whether to add reverse meta-relations
    use_rating_as_edge_weight : store rating as edge_attr on review edges

    Returns
    -------
    HeteroData
    """
    data = HeteroData()

    # ── ID mappings ────────────────────────────────────
    user_map = create_index_mapping(df_reviews["user_id"])
    tire_map = create_index_mapping(df_tires["tire_title"])
    brand_map = create_index_mapping(df_brands["brand"])
    size_map = create_index_mapping(df_sizes["size"])

    num_users = len(user_map)
    num_tires = len(tire_map)
    num_brands = len(brand_map)
    num_sizes = len(size_map)

    # ── Node features / counts ─────────────────────────
    # Users: no content features — store a simple index embedding placeholder
    data["user"].num_nodes = num_users
    data["user"].node_id = torch.arange(num_users)

    # Tires: rich feature vector
    data["tire"].x = torch.from_numpy(tire_features)  # (num_tires, F)
    data["tire"].num_nodes = num_tires

    # Brands & Sizes: identity nodes (no features beyond index)
    data["brand"].num_nodes = num_brands
    data["brand"].node_id = torch.arange(num_brands)

    data["size"].num_nodes = num_sizes
    data["size"].node_id = torch.arange(num_sizes)

    # ── Edge type: (user, reviews, tire) ───────────────
    src_user = df_reviews["user_id"].map(user_map).values
    dst_tire = df_reviews["tire_title"].map(tire_map).values

    # Drop rows where tire_title didn't map (shouldn't happen, but be safe)
    valid = ~(pd.isna(src_user) | pd.isna(dst_tire))
    src_user = src_user[valid].astype(np.int64)
    dst_tire = dst_tire[valid].astype(np.int64)

    edge_index_reviews = torch.tensor(
        np.stack([src_user, dst_tire]), dtype=torch.long
    )
    data["user", "reviews", "tire"].edge_index = edge_index_reviews

    if use_rating_as_edge_weight:
        ratings = df_reviews["rating"].values[valid].astype(np.float32)
        data["user", "reviews", "tire"].edge_attr = torch.from_numpy(
            ratings
        ).unsqueeze(-1)

    if add_reverse_edges:
        data["tire", "rev_by", "user"].edge_index = edge_index_reviews.flip(0)
        if use_rating_as_edge_weight:
            data["tire", "rev_by", "user"].edge_attr = data[
                "user", "reviews", "tire"
            ].edge_attr

    # ── Edge type: (tire, belongs_to, brand) ───────────
    src_t = df_tires["tire_title"].map(tire_map).values.astype(np.int64)
    dst_b = df_tires["brand"].map(brand_map).values.astype(np.int64)

    edge_index_brand = torch.tensor(
        np.stack([src_t, dst_b]), dtype=torch.long
    )
    data["tire", "belongs_to", "brand"].edge_index = edge_index_brand

    if add_reverse_edges:
        data["brand", "has", "tire"].edge_index = edge_index_brand.flip(0)

    # ── Edge type: (tire, has_spec, size) ──────────────
    dst_s = df_tires["size"].map(size_map).values.astype(np.int64)
    edge_index_size = torch.tensor(
        np.stack([src_t, dst_s]), dtype=torch.long
    )
    data["tire", "has_spec", "size"].edge_index = edge_index_size

    if add_reverse_edges:
        data["size", "spec_of", "tire"].edge_index = edge_index_size.flip(0)

    return data


def display_graph_summary(data: HeteroData) -> None:
    """Print a human-readable summary of the heterogeneous graph."""
    print("=" * 60)
    print("  Heterogeneous Graph Summary")
    print("=" * 60)

    print("\nNode types:")
    for node_type in data.node_types:
        store = data[node_type]
        n = store.num_nodes
        feat = ""
        if hasattr(store, "x") and store.x is not None:
            feat = f"  features: {list(store.x.shape)}"
        print(f"  {node_type:>10s} : {n:>6,} nodes{feat}")

    print("\nEdge types:")
    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        name = f"({edge_type[0]}, {edge_type[1]}, {edge_type[2]})"
        attr_info = ""
        if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
            attr_info = f"  edge_attr: {list(data[edge_type].edge_attr.shape)}"
        print(f"  {name:<45s} : {ei.shape[1]:>6,} edges{attr_info}")

    print("=" * 60)
