"""
Bayesian Personalized Ranking (BPR) pairwise loss.

For each (user, positive_tire, negative_tire) triplet, push the
positive's score above the negative's:

    L = -mean(log σ(s(u, t+) - s(u, t-)))
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(score_pos - score_neg).mean()
