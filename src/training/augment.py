"""
Graph augmentations for SGL-style self-supervised contrastive learning.

Two augmented views are produced per training step:
  - Random review-edge dropout (forward + reverse pair, kept consistent).
  - Random tire-feature column dropout on ``data["tire"].x`` (and on
    ``data["tire"].text_x`` if present).

Brand/size edges are NOT dropped — losing them would change the meaning
of those structural relations (a tire still belongs to its brand).
"""

from __future__ import annotations

import copy

import torch
from torch_geometric.data import HeteroData


def augment_view(
    data: HeteroData,
    edge_drop: float = 0.2,
    feat_drop: float = 0.1,
    generator: torch.Generator | None = None,
) -> HeteroData:
    """Return a corrupted copy of ``data`` for the SSL contrastive loss."""
    view = copy.copy(data)  # shallow — we replace the stores we mutate

    fw_key = ("user", "reviews", "tire")
    rv_key = ("tire", "rev_by", "user")

    # MPS generators are not yet supported; sample on the generator's device
    # (typically CPU when running on MPS) and move the mask.
    gen_device = generator.device if generator is not None else None

    # ── Edge dropout (review edges) ──────────────────────────────────
    if edge_drop > 0:
        fw_e = data[fw_key].edge_index
        n_edges = fw_e.size(1)
        sample_device = gen_device if gen_device is not None else fw_e.device
        keep_mask = (
            torch.rand(n_edges, generator=generator, device=sample_device) >= edge_drop
        ).to(fw_e.device)
        # Always keep at least one edge so the encoder has something to pass.
        if keep_mask.sum() == 0:
            keep_mask[0] = True
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]

        fw_store = view[fw_key]
        fw_store.edge_index = fw_e.index_select(1, keep_idx)
        if getattr(fw_store, "edge_attr", None) is not None:
            fw_store.edge_attr = data[fw_key].edge_attr.index_select(0, keep_idx)

        if rv_key in data.edge_types:
            rv_store = view[rv_key]
            rv_store.edge_index = data[rv_key].edge_index.index_select(1, keep_idx)
            if getattr(rv_store, "edge_attr", None) is not None:
                rv_store.edge_attr = data[rv_key].edge_attr.index_select(0, keep_idx)

    # ── Feature dropout (tire spec columns + text columns) ───────────
    if feat_drop > 0:
        tire_store = view["tire"]
        x = data["tire"].x
        sample_device = gen_device if gen_device is not None else x.device
        col_mask = (
            torch.rand(x.size(-1), generator=generator, device=sample_device) >= feat_drop
        ).to(x.device)
        tire_store.x = x * col_mask.float()

        text_x = getattr(data["tire"], "text_x", None)
        if text_x is not None:
            t_sample_device = gen_device if gen_device is not None else text_x.device
            t_mask = (
                torch.rand(text_x.size(-1), generator=generator, device=t_sample_device)
                >= feat_drop
            ).to(text_x.device)
            tire_store.text_x = text_x * t_mask.float()

    return view


def info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Symmetric InfoNCE between two L2-normalised views.

    Both directions are averaged so the gradient is symmetric in v1↔v2.
    """
    z1 = torch.nn.functional.normalize(z1, p=2, dim=-1)
    z2 = torch.nn.functional.normalize(z2, p=2, dim=-1)
    sim = z1 @ z2.t() / temperature                    # (N, N)
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(sim, labels)
        + torch.nn.functional.cross_entropy(sim.t(), labels)
    )
