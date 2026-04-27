"""
HGT encoder for heterogeneous recommendation graphs.

Wraps per-node-type input projections + stacked ``HGTLayer`` blocks to
produce contextualised embeddings for every node type in the graph.

Input projection strategy:
    - Node types with a feature matrix ``data[nt].x``  → ``nn.Linear``
    - Featureless node types                           → ``nn.Embedding``
      (indexed by ``data[nt].node_id``; falls back to ``arange(num_nodes)``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.models.hgt_layer import EdgeType, HGTLayer


class HGTEncoder(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[EdgeType]],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_norm: bool = True,
        aggregate_layers: str = "mean",
        in_dim_dict: dict[str, int] | None = None,
        num_nodes_dict: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if aggregate_layers not in {"last", "mean"}:
            raise ValueError(
                "aggregate_layers must be either 'last' or 'mean', "
                f"got {aggregate_layers!r}"
            )
        node_types, edge_types = metadata

        in_dim_dict = in_dim_dict or {}
        num_nodes_dict = num_nodes_dict or {}

        self.hidden_dim = hidden_dim
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.aggregate_layers = aggregate_layers

        self.input_proj = nn.ModuleDict()
        self.input_emb = nn.ModuleDict()
        for nt in self.node_types:
            if nt in in_dim_dict:
                self.input_proj[nt] = nn.Linear(in_dim_dict[nt], hidden_dim)
            elif nt in num_nodes_dict:
                self.input_emb[nt] = nn.Embedding(num_nodes_dict[nt], hidden_dim)
            else:
                raise ValueError(
                    f"Node type '{nt}' has neither an input dim nor a node count — "
                    f"cannot build input projection."
                )

        self.layers = nn.ModuleList(
            HGTLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                node_types=self.node_types,
                edge_types=self.edge_types,
                num_heads=num_heads,
                dropout=dropout,
                use_norm=use_norm,
            )
            for _ in range(num_layers)
        )

    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_norm: bool = True,
        aggregate_layers: str = "mean",
    ) -> "HGTEncoder":
        """Build an encoder configured to match a ``HeteroData`` instance."""
        in_dim_dict: dict[str, int] = {}
        num_nodes_dict: dict[str, int] = {}
        for nt in data.node_types:
            store = data[nt]
            if getattr(store, "x", None) is not None:
                in_dim_dict[nt] = store.x.size(-1)
            else:
                num_nodes_dict[nt] = store.num_nodes

        return cls(
            metadata=data.metadata(),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_norm=use_norm,
            aggregate_layers=aggregate_layers,
            in_dim_dict=in_dim_dict,
            num_nodes_dict=num_nodes_dict,
        )

    # ──────────────────────────────────────────────────────────────────
    def _initial_embeddings(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            store = data[nt]
            if nt in self.input_proj:
                h[nt] = self.input_proj[nt](store.x)
            else:
                emb = self.input_emb[nt]
                ids = getattr(store, "node_id", None)
                if ids is None:
                    ids = torch.arange(store.num_nodes, device=emb.weight.device)
                h[nt] = emb(ids.to(emb.weight.device))
        return h

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h_dict = self._initial_embeddings(data)
        layer_outputs = [h_dict]
        edge_index_dict = data.edge_index_dict
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            layer_outputs.append(h_dict)

        if self.aggregate_layers == "last":
            return h_dict

        return {
            nt: torch.stack([h[nt] for h in layer_outputs], dim=0).mean(dim=0)
            for nt in h_dict
        }
