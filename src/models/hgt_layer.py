"""
Heterogeneous Graph Transformer (HGT) layer.

Faithful implementation of the layer described in Hu et al., 2020
(https://arxiv.org/abs/2003.01332), Section 3.

Per meta-relation ⟨τ(s), ϕ(e), τ(t)⟩ the layer computes:
    Attention:  Eq. (3)   ATT-head_i(s,e,t) =
                          ( K_i(s) · W^ATT_{ϕ(e)} · Q_i(t)^T ) · μ_{⟨τ(s),ϕ(e),τ(t)⟩} / √d
    Message:    Eq. (4)   MSG-head_i(s,e,t) = M_i(s) · W^MSG_{ϕ(e)}
    Update:     Eq. (5)   H^(l)[t] = A-Linear_{τ(t)}( σ( H̃^(l)[t] ) ) + H^(l-1)[t]

Parameter sharing:
    K-/Q-/M-/A-Linear       → indexed by node type τ
    W^ATT, W^MSG            → indexed by edge type ϕ
    μ  (prior scaling)      → indexed by full triplet ⟨τ(s), ϕ(e), τ(t)⟩
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


EdgeType = tuple[str, str, str]


class HGTLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_types: list[str],
        edge_types: list[EdgeType],
        num_heads: int = 4,
        dropout: float = 0.2,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.use_norm = use_norm

        # ── Per-node-type linear projections (K, Q, M, A) ───────────────
        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.m_lin = nn.ModuleDict()
        self.a_lin = nn.ModuleDict()
        self.norms = nn.ModuleDict() if use_norm else None
        for nt in self.node_types:
            self.k_lin[nt] = nn.Linear(in_dim, out_dim)
            self.q_lin[nt] = nn.Linear(in_dim, out_dim)
            self.m_lin[nt] = nn.Linear(in_dim, out_dim)
            self.a_lin[nt] = nn.Linear(out_dim, out_dim)
            if use_norm:
                self.norms[nt] = nn.LayerNorm(out_dim)

        # ── Per-edge-type relation matrices (W^ATT, W^MSG) ──────────────
        # One (d_k × d_k) matrix per head per edge type.
        self.w_att = nn.ParameterDict()
        self.w_msg = nn.ParameterDict()
        # ── Per-meta-relation prior scalar μ (one per head) ─────────────
        self.mu = nn.ParameterDict()
        for et in self.edge_types:
            key = self._edge_key(et)
            self.w_att[key] = nn.Parameter(torch.empty(num_heads, self.d_k, self.d_k))
            self.w_msg[key] = nn.Parameter(torch.empty(num_heads, self.d_k, self.d_k))
            self.mu[key] = nn.Parameter(torch.ones(num_heads))
            nn.init.xavier_uniform_(self.w_att[key])
            nn.init.xavier_uniform_(self.w_msg[key])

        # ── Gated residual (learnable α per target node type) ──────────
        # Follows the reference pyHGT implementation: H = α·new + (1-α)·H_prev.
        # Paper writes plain residual; gated form generalises it and helps
        # when the aggregated signal is weak (e.g., featureless node types).
        self.skip = nn.ParameterDict(
            {nt: nn.Parameter(torch.ones(1)) for nt in self.node_types}
        )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _edge_key(edge_type: EdgeType) -> str:
        return "__".join(edge_type)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        H = self.num_heads
        Dk = self.d_k

        # Project every node type once into K, Q, M spaces (Eq. 3 / 4 first step).
        K = {nt: self.k_lin[nt](x).view(-1, H, Dk) for nt, x in x_dict.items()}
        Q = {nt: self.q_lin[nt](x).view(-1, H, Dk) for nt, x in x_dict.items()}
        M = {nt: self.m_lin[nt](x).view(-1, H, Dk) for nt, x in x_dict.items()}

        # Per-target-type buckets: collect (att_logits, message, dst_idx)
        # across every incoming edge type, then softmax once per target node.
        buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            nt: [] for nt in self.node_types
        }

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if edge_index.numel() == 0:
                continue
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            key = self._edge_key(edge_type)
            src, dst = edge_index[0], edge_index[1]

            k_s = K[src_type][src]   # (E, H, Dk)
            q_t = Q[dst_type][dst]   # (E, H, Dk)
            m_s = M[src_type][src]   # (E, H, Dk)

            # Attention: k · W^ATT · q^T, per head.
            # k_s: (E, H, Dk), w_att: (H, Dk, Dk)  →  k_trans: (E, H, Dk)
            k_trans = torch.einsum("ehd,hdf->ehf", k_s, self.w_att[key])
            att_logits = (k_trans * q_t).sum(dim=-1)             # (E, H)
            att_logits = att_logits * self.mu[key] / math.sqrt(Dk)

            # Message: m · W^MSG, per head.  (E, H, Dk) × (H, Dk, Dk) → (E, H, Dk)
            msg = torch.einsum("ehd,hdf->ehf", m_s, self.w_msg[key])

            buckets[dst_type].append((att_logits, msg, dst))

        out_dict: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            x_prev = x_dict.get(nt)
            if x_prev is None:
                continue

            items = buckets[nt]
            if not items:
                # No incoming edges — keep the previous representation.
                out_dict[nt] = x_prev
                continue

            att = torch.cat([a for a, _, _ in items], dim=0)      # (ΣE, H)
            msg = torch.cat([m for _, m, _ in items], dim=0)      # (ΣE, H, Dk)
            dst = torch.cat([d for _, _, d in items], dim=0)      # (ΣE,)

            # Softmax over *all* incoming edges per target node, regardless
            # of edge type — this is Eq. 3's outer softmax across N(t).
            num_dst = x_prev.size(0)
            att = softmax(att, dst, num_nodes=num_dst)            # (ΣE, H)

            # Weighted sum of messages -> aggregated pre-update vector.
            weighted = msg * att.unsqueeze(-1)                    # (ΣE, H, Dk)
            agg = weighted.new_zeros(num_dst, H, Dk)
            agg.index_add_(0, dst, weighted)
            agg = agg.reshape(num_dst, self.out_dim)

            # Eq. 5: A-Linear applied after non-linearity + gated residual.
            new = self.a_lin[nt](F.gelu(agg))
            new = self.dropout(new)

            alpha = torch.sigmoid(self.skip[nt])
            out = alpha * new + (1 - alpha) * x_prev
            if self.use_norm:
                out = self.norms[nt](out)
            out_dict[nt] = out

        # Preserve types that were passed in but never touched (e.g., isolated).
        for nt, x in x_dict.items():
            out_dict.setdefault(nt, x)

        return out_dict
