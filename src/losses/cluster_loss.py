"""
Auxiliary CE loss for DeepCluster (Path A).

Inverse-frequency weighting prevents the trivial-parametrization
collapse described in DeepCluster §3.3 — without it, a few large
clusters dominate the gradient and the encoder degenerates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def deep_cluster_loss(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    num_clusters: int,
    use_inverse_freq_weight: bool = True,
) -> torch.Tensor:
    if not use_inverse_freq_weight:
        return F.cross_entropy(logits, pseudo_labels)

    counts = torch.bincount(pseudo_labels, minlength=num_clusters).float().clamp(min=1.0)
    weights = (counts.sum() / counts) / num_clusters
    return F.cross_entropy(logits, pseudo_labels, weight=weights.to(logits.device))
