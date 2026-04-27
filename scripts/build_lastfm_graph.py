#!/usr/bin/env python3
"""
Build a LastFM heterogeneous recommendation graph.

The PyG LastFM dataset contains user, artist, and tag nodes. This builder maps
``artist`` to the canonical ``item`` node type used by the recommender:

    user --reviews--> item(artist)
    item --rev_by--> user
    user --follows--> user
    item --has_tag--> tag
    tag  --tag_of--> item
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import LastFM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.graph_utils import display_graph_summary


def build_graph(root: Path) -> dict[str, object]:
    source = LastFM(str(root))[0]
    data = HeteroData()

    data["user"].num_nodes = source["user"].num_nodes
    data["user"].node_id = torch.arange(source["user"].num_nodes)
    data["item"].num_nodes = source["artist"].num_nodes
    data["item"].node_id = torch.arange(source["artist"].num_nodes)
    data["tag"].num_nodes = source["tag"].num_nodes
    data["tag"].node_id = torch.arange(source["tag"].num_nodes)

    review_edges = source["user", "to", "artist"].edge_index.long()
    review_edge_id = torch.arange(review_edges.size(1), dtype=torch.long)
    ratings = torch.full((review_edges.size(1), 1), 5.0, dtype=torch.float32)
    data["user", "reviews", "item"].edge_index = review_edges
    data["user", "reviews", "item"].edge_attr = ratings
    data["user", "reviews", "item"].review_edge_id = review_edge_id
    data["item", "rev_by", "user"].edge_index = review_edges.flip(0)
    data["item", "rev_by", "user"].edge_attr = ratings
    data["item", "rev_by", "user"].review_edge_id = review_edge_id

    if ("user", "to", "user") in source.edge_types:
        data["user", "follows", "user"].edge_index = source[
            "user", "to", "user"
        ].edge_index.long()

    if ("artist", "to", "tag") in source.edge_types:
        data["item", "has_tag", "tag"].edge_index = source[
            "artist", "to", "tag"
        ].edge_index.long()
    if ("tag", "to", "artist") in source.edge_types:
        data["tag", "tag_of", "item"].edge_index = source[
            "tag", "to", "artist"
        ].edge_index.long()

    return {
        "graph": data,
        "mappings": {
            "user_map": {i: i for i in range(data["user"].num_nodes)},
            "item_map": {i: i for i in range(data["item"].num_nodes)},
            "artist_map": {i: i for i in range(data["item"].num_nodes)},
            "tag_map": {i: i for i in range(data["tag"].num_nodes)},
        },
        "source": "pyg_lastfm",
    }


def sanity_check(payload: dict[str, object]) -> None:
    data = payload["graph"]
    assert isinstance(data, HeteroData)
    assert data["user", "reviews", "item"].edge_index.size(1) > 0
    for ntype in data.node_types:
        assert data[ntype].num_nodes > 0
    for etype in data.edge_types:
        edge_index = data[etype].edge_index
        src_type, _, dst_type = etype
        assert edge_index.shape[0] == 2
        assert edge_index.min() >= 0
        assert edge_index[0].max() < data[src_type].num_nodes
        assert edge_index[1].max() < data[dst_type].num_nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LastFM HGT graph.")
    parser.add_argument(
        "--root",
        default="data/raw/lastfm",
        help="Where PyG should download/process LastFM.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/lastfm_hetero_graph.pt",
        help="Output graph payload path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_graph(PROJECT_ROOT / args.root)
    data = payload["graph"]
    display_graph_summary(data)
    sanity_check(payload)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    loaded = torch.load(output_path, weights_only=False)
    assert loaded["graph"].node_types == data.node_types
    print(f"\nSaved LastFM graph payload to {output_path}")
    print("Reload verification passed.")


if __name__ == "__main__":
    main()
