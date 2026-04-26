"""Generic graph utilities for HeteroData payloads."""

from __future__ import annotations

from torch_geometric.data import HeteroData


def display_graph_summary(data: HeteroData) -> None:
    """Print a human-readable summary of a heterogeneous graph."""
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
        edge_index = data[edge_type].edge_index
        name = f"({edge_type[0]}, {edge_type[1]}, {edge_type[2]})"
        attr = getattr(data[edge_type], "edge_attr", None)
        attr_info = f"  edge_attr: {list(attr.shape)}" if attr is not None else ""
        print(f"  {name:<45s} : {edge_index.shape[1]:>6,} edges{attr_info}")

    print("=" * 60)
