#!/usr/bin/env python3
"""
Smoke test: load the saved heterogeneous graph, build an HGT encoder,
run one forward pass, and print the output embedding shapes.

Usage
-----
    uv run python tests/test_hgt_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import HGTEncoder


def main() -> None:
    graph_path = PROJECT_ROOT / "data" / "processed" / "hetero_graph.pt"
    data = torch.load(graph_path, weights_only=False)["graph"]

    model = HGTEncoder.from_data(data, hidden_dim=128, num_layers=2, num_heads=4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    model.eval()
    with torch.no_grad():
        out = model(data)
    for nt, h in out.items():
        print(f"  {nt:>6s}: {tuple(h.shape)}")


if __name__ == "__main__":
    main()
