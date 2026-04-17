#!/usr/bin/env python3
"""
Visualize the heterogeneous tire graph.

Generates several plots:
  1. Schema diagram  -- node types and edge types as a meta-graph
  2. Subgraph sample -- a small sampled neighbourhood for inspection
  3. Degree distributions per node type
  4. Tire feature distributions (histograms)

Usage
-----
    uv run python scripts/visualize_graph.py
    uv run python scripts/visualize_graph.py --graph data/processed/hetero_graph.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Colour palette for node types ─────────────────────────
NODE_COLORS = {
    "user":  "#4A90D9",   # blue
    "tire":  "#E07B53",   # orange
    "brand": "#6DBE6D",   # green
    "size":  "#C27ABA",   # purple
}

EDGE_COLORS = {
    "reviews":    "#7BAFD4",
    "rev_by":     "#7BAFD4",
    "belongs_to": "#A0D4A0",
    "has":        "#A0D4A0",
    "has_spec":   "#D4A0CC",
    "spec_of":    "#D4A0CC",
}


def plot_schema(data, save_path: Path) -> None:
    """Draw the meta-graph: node types as boxes, edge types as arrows."""
    G = nx.MultiDiGraph()

    for ntype in data.node_types:
        G.add_node(ntype)

    for src, rel, dst in data.edge_types:
        G.add_edge(src, dst, label=rel)

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = {
        "user":  (0.0, 1.0),
        "tire":  (1.0, 1.0),
        "brand": (2.0, 1.5),
        "size":  (2.0, 0.5),
    }

    # Draw nodes
    for ntype, (x, y) in pos.items():
        color = NODE_COLORS.get(ntype, "#CCCCCC")
        count = data[ntype].num_nodes
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.25, y - 0.15), 0.5, 0.3,
            facecolor=color, edgecolor="white", linewidth=2, zorder=3,
            alpha=0.9, boxstyle="round,pad=0.02",
        ))
        ax.text(x, y + 0.02, ntype, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.08, f"({count:,})", ha="center", va="center",
                fontsize=9, color="white", alpha=0.85, zorder=4)

    # Draw edges with curved arrows
    drawn = set()
    for src, rel, dst in data.edge_types:
        pair_key = tuple(sorted([src, dst])) + (rel,)
        if pair_key in drawn:
            continue
        drawn.add(pair_key)

        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_count = data[src, rel, dst].edge_index.shape[1]
        color = EDGE_COLORS.get(rel, "#999999")

        ax.annotate(
            "", xy=(x1 - 0.25, y1), xytext=(x0 + 0.25, y0),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.8,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=2,
        )
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.12
        ax.text(mx, my, f"{rel}\n({edge_count:,})", ha="center", va="center",
                fontsize=8, color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", alpha=0.85))

    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(0.1, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Heterogeneous Graph Schema", fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Schema diagram saved to {save_path}")
    plt.close(fig)


def plot_subgraph_sample(data, save_path: Path, num_users: int = 5) -> None:
    """Sample a small subgraph around a few users and visualize it."""
    G = nx.Graph()

    # Pick random users
    rng = np.random.default_rng(42)
    user_indices = rng.choice(data["user"].num_nodes, size=num_users, replace=False)

    # Get review edges
    ei = data["user", "reviews", "tire"].edge_index.numpy()
    sampled_tires = set()
    for uid in user_indices:
        mask = ei[0] == uid
        tires = ei[1][mask]
        G.add_node(f"U{uid}", ntype="user")
        for tid in tires:
            G.add_node(f"T{tid}", ntype="tire")
            G.add_edge(f"U{uid}", f"T{tid}")
            sampled_tires.add(tid)

    # Add brand/size edges for sampled tires
    if ("tire", "belongs_to", "brand") in data.edge_types:
        ei_brand = data["tire", "belongs_to", "brand"].edge_index.numpy()
        for tid in sampled_tires:
            mask = ei_brand[0] == tid
            for bid in ei_brand[1][mask]:
                G.add_node(f"B{bid}", ntype="brand")
                G.add_edge(f"T{tid}", f"B{bid}")

    if ("tire", "has_spec", "size") in data.edge_types:
        ei_size = data["tire", "has_spec", "size"].edge_index.numpy()
        for tid in sampled_tires:
            mask = ei_size[0] == tid
            for sid in ei_size[1][mask]:
                G.add_node(f"S{sid}", ntype="size")
                G.add_edge(f"T{tid}", f"S{sid}")

    # Colour nodes by type
    colors = []
    for node in G.nodes():
        ntype = G.nodes[node].get("ntype", "tire")
        colors.append(NODE_COLORS.get(ntype, "#CCCCCC"))

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=120, alpha=0.85)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CCCCCC", alpha=0.5, width=0.8)

    # Legend
    handles = [mpatches.Patch(color=c, label=t) for t, c in NODE_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_title(
        f"Subgraph sample ({num_users} users, {len(sampled_tires)} tires)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Subgraph sample saved to {save_path}")
    plt.close(fig)


def plot_degree_distributions(data, save_path: Path) -> None:
    """Plot degree distribution for each edge type's source nodes."""
    edge_types_to_plot = [
        ("user", "reviews", "tire"),
        ("tire", "belongs_to", "brand"),
        ("tire", "has_spec", "size"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, etype in zip(axes, edge_types_to_plot):
        src_type, rel, dst_type = etype
        if etype not in data.edge_types:
            continue

        ei = data[etype].edge_index
        src_indices = ei[0].numpy()
        num_src = data[src_type].num_nodes

        # Count degree per source node
        degrees = np.bincount(src_indices, minlength=num_src)

        ax.hist(degrees, bins=min(50, int(degrees.max()) + 1),
                color=NODE_COLORS.get(src_type, "#999"), alpha=0.8, edgecolor="white")
        ax.set_xlabel("Degree", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{src_type} --[{rel}]--> {dst_type}", fontsize=11, fontweight="bold")

        # Add stats annotation
        ax.text(0.95, 0.95,
                f"mean: {degrees.mean():.1f}\nmax: {degrees.max()}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.8))

    fig.suptitle("Degree Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Degree distributions saved to {save_path}")
    plt.close(fig)


def plot_tire_features(data, save_path: Path) -> None:
    """Histogram of each tire feature dimension."""
    features = data["tire"].x.numpy()
    num_feats = features.shape[1]
    feat_names = [
        "price", "avg_rating", "rating_count", "treadwear",
        "traction", "temperature", "speed_rating",
    ]
    # Extend names if needed
    while len(feat_names) < num_feats:
        feat_names.append(f"feature_{len(feat_names)}")

    cols = min(4, num_feats)
    rows = (num_feats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for i in range(num_feats):
        ax = axes[i]
        ax.hist(features[:, i], bins=50, color=NODE_COLORS["tire"], alpha=0.8, edgecolor="white")
        ax.set_title(feat_names[i], fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=9)

    # Hide unused axes
    for i in range(num_feats, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Tire Node Feature Distributions (after scaling)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Feature distributions saved to {save_path}")
    plt.close(fig)


def main(graph_path: str = "data/processed/hetero_graph.pt") -> None:
    full_path = PROJECT_ROOT / graph_path
    print(f"Loading graph from {full_path} ...")
    data = torch.load(full_path, weights_only=False)

    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations ...")
    plot_schema(data, output_dir / "graph_schema.png")
    plot_subgraph_sample(data, output_dir / "subgraph_sample.png")
    plot_degree_distributions(data, output_dir / "degree_distributions.png")
    plot_tire_features(data, output_dir / "tire_feature_distributions.png")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the heterogeneous tire graph")
    parser.add_argument(
        "--graph", default="data/processed/hetero_graph.pt",
        help="Path to the saved HeteroData .pt file",
    )
    args = parser.parse_args()
    main(graph_path=args.graph)
