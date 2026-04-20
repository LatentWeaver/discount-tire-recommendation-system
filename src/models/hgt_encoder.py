"""
HGT encoder for the tire recommendation graph.

Wraps per-node-type input projections + stacked ``HGTLayer`` blocks to
produce contextualised embeddings for every node type in the graph.

Input projection strategy:
    - Node types with a feature matrix ``data[nt].x``  → ``nn.Linear``
    - Featureless node types with shared seeds (user by default)
      → one learnable vector repeated across all nodes of that type
    - Other featureless node types (brand / size)       → ``nn.Embedding``
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
        in_dim_dict: dict[str, int] | None = None,
        num_nodes_dict: dict[str, int] | None = None,
        shared_seed_node_types: tuple[str, ...] = ("user",),
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata

        in_dim_dict = in_dim_dict or {}
        num_nodes_dict = num_nodes_dict or {}
        self.shared_seed_node_types = set(shared_seed_node_types)

        self.hidden_dim = hidden_dim
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        self.input_proj = nn.ModuleDict()
        self.input_emb = nn.ModuleDict()
        self.type_seed = nn.ParameterDict()
        for nt in self.node_types:
            if nt in in_dim_dict:
                self.input_proj[nt] = nn.Linear(in_dim_dict[nt], hidden_dim)
            elif nt in self.shared_seed_node_types:
                self.type_seed[nt] = nn.Parameter(torch.empty(hidden_dim))
                nn.init.normal_(self.type_seed[nt], std=0.02)
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
        shared_seed_node_types: tuple[str, ...] = ("user",),
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
            in_dim_dict=in_dim_dict,
            num_nodes_dict=num_nodes_dict,
            shared_seed_node_types=shared_seed_node_types,
        )

    # ──────────────────────────────────────────────────────────────────
    def _initial_embeddings(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            store = data[nt]
            if nt in self.input_proj:
                h[nt] = self.input_proj[nt](store.x)
            elif nt in self.type_seed:
                h[nt] = self.type_seed[nt].unsqueeze(0).expand(store.num_nodes, -1)
            else:
                emb = self.input_emb[nt]
                ids = getattr(store, "node_id", None)
                if ids is None:
                    ids = torch.arange(store.num_nodes, device=emb.weight.device)
                h[nt] = emb(ids.to(emb.weight.device))
        return h

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h_dict = self._initial_embeddings(data)
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr
            for edge_type in data.edge_types
            if getattr(data[edge_type], "edge_attr", None) is not None
        }
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        return h_dict
