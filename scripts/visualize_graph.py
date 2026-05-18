#!/usr/bin/env python3
"""
Visualize the LightGCN Yelp 2018 heterogeneous graph.

Three figures land in ``outputs/figures/``:

  graph_schema.png       — node/edge-type diagram (HGT slot view)
  degree_distributions.png — user + item interaction-count CCDFs (log–log)
  subgraph_sample.png    — a bipartite subgraph sample of N users and the
                           items they interacted with in the train split

Only TRAIN edges drive the sample so the picture matches what the encoder
sees during message passing (val/test edges are held out and never visible
to the model).

Usage
-----
    uv run python scripts/visualize_graph.py
    uv run python scripts/visualize_graph.py --num-users 12 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _train_user_item_edges(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (user_idx, item_idx) int arrays restricted to the train slice."""
    data = payload["graph"]
    edge_index = data["user", "reviews", "tire"].edge_index.cpu().numpy()
    split = payload.get("precomputed_split")
    if split is None:
        return edge_index[0], edge_index[1]
    train_idx = split["train_idx"].cpu().numpy()
    return edge_index[0, train_idx], edge_index[1, train_idx]


# ── 1. Schema diagram ────────────────────────────────────────────────
def plot_schema(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    nodes = {
        "user":  (1.2, 3.0, "#4C72B0"),
        "tire":  (5.0, 3.0, "#DD8452"),
        "brand": (8.4, 4.4, "#55A868"),
        "size":  (8.4, 1.6, "#C44E52"),
    }
    labels = {
        "user":  "user\n(31 668)",
        "tire":  "item\n(38 048)",
        "brand": "brand\n(1 — ALL)",
        "size":  "size\n(1 — ALL)",
    }
    for k, (x, y, c) in nodes.items():
        ax.add_patch(FancyBboxPatch(
            (x - 0.7, y - 0.45), 1.4, 0.9,
            boxstyle="round,pad=0.05", linewidth=1.4,
            edgecolor=c, facecolor=c, alpha=0.25,
        ))
        ax.text(x, y, labels[k], ha="center", va="center", fontsize=11, fontweight="bold")

    def arrow(src: str, dst: str, label: str, offset: tuple[float, float] = (0.0, 0.18)) -> None:
        x1, y1, _ = nodes[src]
        x2, y2, _ = nodes[dst]
        ax.add_patch(FancyArrowPatch(
            (x1 + 0.7, y1), (x2 - 0.7, y2),
            arrowstyle="-|>", mutation_scale=18,
            color="#333", linewidth=1.2,
        ))
        mx, my = (x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=9,
                style="italic", color="#333")

    arrow("user", "tire", "reviews")
    arrow("tire", "brand", "belongs_to")
    arrow("tire", "size",  "has_spec")

    ax.set_title("LightGCN Yelp 2018 — HGT Graph Schema",
                 fontsize=13, fontweight="bold", pad=14)
    ax.text(5, 0.3,
            "Implicit feedback only — every (user→item) edge is a positive interaction.\n"
            "brand & size collapse to a single ALL node (LightGCN release has no item categories).",
            ha="center", va="center", fontsize=9, color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── 2. Degree distributions ──────────────────────────────────────────
def plot_degree_distributions(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    num_users: int,
    num_items: int,
    out_path: Path,
) -> None:
    user_deg = np.bincount(user_idx, minlength=num_users)
    item_deg = np.bincount(item_idx, minlength=num_items)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, deg, title, color in [
        (axes[0], user_deg, "Interactions per user", "#4C72B0"),
        (axes[1], item_deg, "Interactions per item", "#DD8452"),
    ]:
        deg_nonzero = deg[deg > 0]
        bins = np.logspace(0, np.log10(max(deg_nonzero.max(), 2)), 40)
        ax.hist(deg_nonzero, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(
            f"{title}\nn={deg_nonzero.size:,}  ·  "
            f"min={deg_nonzero.min()}  ·  median={int(np.median(deg_nonzero))}  ·  "
            f"max={deg_nonzero.max()}",
            fontsize=10,
        )
        ax.set_xlabel("degree (train edges)")
        ax.set_ylabel("count of nodes")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Yelp 2018 — degree distributions (train split)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── 3. Subgraph sample (bipartite) ───────────────────────────────────
def plot_subgraph_sample(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    num_users: int,
    n_sample_users: int,
    seed: int,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(seed)
    # Sample users that have at least a couple of train interactions so the
    # picture isn't dominated by hairline strands.
    user_deg = np.bincount(user_idx, minlength=num_users)
    eligible = np.where(user_deg >= 3)[0]
    if eligible.size == 0:
        raise RuntimeError("No users with ≥3 train interactions — graph too sparse to sample.")
    sampled_users = rng.choice(eligible, size=min(n_sample_users, eligible.size), replace=False)

    mask = np.isin(user_idx, sampled_users)
    sub_u = user_idx[mask]
    sub_t = item_idx[mask]
    sampled_items = np.unique(sub_t)

    g = nx.Graph()
    for u in sampled_users:
        g.add_node(("u", int(u)), bipartite=0)
    for t in sampled_items:
        g.add_node(("t", int(t)), bipartite=1)
    for u, t in zip(sub_u, sub_t):
        g.add_edge(("u", int(u)), ("t", int(t)))

    user_nodes = [n for n in g.nodes if n[0] == "u"]
    item_nodes = [n for n in g.nodes if n[0] == "t"]
    n_u, n_i = len(user_nodes), len(item_nodes)

    # Spring layout — users repel each other while items cluster around
    # whichever users they connect to. Shared items end up between users
    # and isolated items get pushed to the periphery.
    pos = nx.spring_layout(g, k=1.6 / np.sqrt(max(g.number_of_nodes(), 1)),
                           iterations=200, seed=seed)

    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(g, pos, alpha=0.18, width=0.45, ax=ax)
    # Items first (smaller, behind), users second (larger, in front).
    nx.draw_networkx_nodes(g, pos, nodelist=item_nodes,
                           node_color="#DD8452", node_size=28,
                           edgecolors="white", linewidths=0.3, ax=ax,
                           label="item")
    nx.draw_networkx_nodes(g, pos, nodelist=user_nodes,
                           node_color="#4C72B0", node_size=240,
                           edgecolors="white", linewidths=1.0, ax=ax,
                           label="user")

    nx.draw_networkx_labels(
        g, pos,
        labels={n: f"u{n[1]}" for n in user_nodes},
        font_size=8, font_color="white", font_weight="bold",
        ax=ax,
    )

    ax.set_axis_off()
    ax.set_title(
        f"Yelp 2018 — random subgraph sample of {n_u} users "
        f"({len(sub_u):,} train edges → {n_i:,} unique items)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9, markerscale=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main(graph_path: str, num_users: int, seed: int) -> None:
    full = PROJECT_ROOT / graph_path
    print(f"Loading {full} ...")
    payload = torch.load(full, weights_only=False)
    data = payload["graph"]
    n_users = int(data["user"].num_nodes)
    n_items = int(data["tire"].num_nodes)
    user_idx, item_idx = _train_user_item_edges(payload)
    print(
        f"  Users: {n_users:,}  ·  Items: {n_items:,}  ·  "
        f"Train edges: {user_idx.size:,}"
    )

    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Writing figures ...")
    plot_schema(out_dir / "graph_schema.png")
    plot_degree_distributions(user_idx, item_idx, n_users, n_items,
                              out_dir / "degree_distributions.png")
    plot_subgraph_sample(user_idx, item_idx, n_users, num_users, seed,
                         out_dir / "subgraph_sample.png")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the Yelp 2018 hetero graph")
    parser.add_argument("--graph", default="data/processed/hetero_graph_yelp2018.pt",
                        help="Path (relative to project root) to the .pt payload.")
    parser.add_argument("--num-users", type=int, default=10,
                        help="Number of users in the bipartite subgraph sample.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the user sample.")
    args = parser.parse_args()
    main(graph_path=args.graph, num_users=args.num_users, seed=args.seed)
