"""
Hard-negative mining within (brand ∪ size) buckets.

For each tire ``t``, the hard-negative candidate pool is the union of
- other tires sharing ``t``'s brand
- other tires sharing ``t``'s size

stored as a CSR-style (offsets, indices) pair so per-tire sampling is a
single randint into a contiguous slice on GPU.

Why brand ∪ size: random negatives are trivially different from a
positive (different size, different brand). The model only learns to
separate "235/40R18 Michelin" from "P175/70R14 Cooper" — easy. Hard
negatives within the same brand or size force the model to discriminate
among substitutes, which is what real retrieval needs to do.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def build_buckets(
    brand_idx: torch.Tensor,
    size_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (offsets, indices) for the (brand ∪ size) hard-neg pool per tire.

    Parameters
    ----------
    brand_idx : (N_tire,) brand node id of each tire
    size_idx  : (N_tire,) size  node id of each tire

    Returns
    -------
    offsets : (N_tire + 1,) long — CSR offsets into ``indices``
    indices : (M,) long — concatenated bucket members per tire (excluding self)
    """
    n_tire = brand_idx.size(0)
    device = brand_idx.device

    tires_in_brand: dict[int, list[int]] = defaultdict(list)
    tires_in_size: dict[int, list[int]] = defaultdict(list)
    for t, b in enumerate(brand_idx.tolist()):
        tires_in_brand[b].append(t)
    for t, s in enumerate(size_idx.tolist()):
        tires_in_size[s].append(t)

    flat: list[int] = []
    offsets: list[int] = [0]
    for t in range(n_tire):
        b = int(brand_idx[t])
        s = int(size_idx[t])
        peers = set(tires_in_brand[b]) | set(tires_in_size[s])
        peers.discard(t)
        flat.extend(peers)
        offsets.append(len(flat))

    return (
        torch.tensor(offsets, dtype=torch.long, device=device),
        torch.tensor(flat, dtype=torch.long, device=device),
    )


def sample_hard_negatives(
    pos_tires: torch.Tensor,
    k_hard: int,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    num_tires: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``k_hard`` hard negatives per positive tire.

    Empty buckets fall back to uniform-random tires so the output shape
    is always (B, k_hard).
    """
    device = offsets.device
    B = pos_tires.size(0)

    starts = offsets[pos_tires]                 # (B,)
    ends = offsets[pos_tires + 1]               # (B,)
    lengths = (ends - starts).clamp(min=1)      # avoid div-by-zero — cold rows replaced below

    # Random offset into each bucket: 0..lengths[i]-1.
    rand = torch.rand((B, k_hard), generator=generator, device=device)
    offs = (rand * lengths.unsqueeze(-1).float()).long()
    offs = offs.clamp(max=lengths.unsqueeze(-1) - 1)
    flat_pos = starts.unsqueeze(-1) + offs       # (B, k_hard)
    hard = indices[flat_pos]                     # (B, k_hard)

    # Cold buckets (true length == 0): fall back to random tire.
    true_lengths = ends - starts
    cold = (true_lengths == 0).unsqueeze(-1).expand_as(hard)
    if cold.any():
        rand_tires = torch.randint(
            0, num_tires, (B, k_hard), generator=generator, device=device
        )
        hard = torch.where(cold, rand_tires, hard)

    return hard
