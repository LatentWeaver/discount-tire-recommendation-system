"""
Offline DeepCluster pseudo-label refresh.

Called every N epochs from the training loop. Pipeline:

    h_tire (detached, CPU) ──► PCA ──► whiten ──► ℓ2-norm ──► k-means
                                                              │
                                                              ▼
                                              pseudo-labels  ŷ_tire
                                              (with empty-cluster repair)

The pseudo-labels are then frozen and used as classification targets for
``ClusterHead`` until the next refresh.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def refresh_pseudo_labels(
    h_tire: torch.Tensor,
    num_clusters: int,
    pca_dim: int | None = 64,
    n_init: int = 10,
    max_iter: int = 100,
    seed: int = 0,
    empty_cluster_eps: float = 1e-3,
) -> torch.Tensor:
    """Run the full DeepCluster preprocessing + k-means and return labels."""
    x = h_tire.detach().cpu().numpy().astype(np.float32)
    n, d = x.shape

    if pca_dim is not None and pca_dim < d:
        x = PCA(n_components=pca_dim, whiten=True, random_state=seed).fit_transform(x)
    else:
        x = x - x.mean(axis=0, keepdims=True)
        x = x / (x.std(axis=0, keepdims=True) + 1e-8)

    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    km = KMeans(
        n_clusters=num_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    labels = km.fit_predict(x).astype(np.int64)
    centroids = km.cluster_centers_.copy()

    labels = _repair_empty_clusters(
        x=x,
        labels=labels,
        centroids=centroids,
        num_clusters=num_clusters,
        eps=empty_cluster_eps,
        seed=seed,
    )
    return torch.from_numpy(labels)


def _repair_empty_clusters(
    x: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    num_clusters: int,
    eps: float,
    seed: int,
) -> np.ndarray:
    """
    Reassign empty clusters by perturbing a non-empty centroid and
    splitting half of its points to the new centroid. Mirrors the trick
    described in DeepCluster §3.3.
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=num_clusters)

    for k in np.where(counts == 0)[0]:
        donors = np.where(counts > 1)[0]
        if donors.size == 0:
            break
        donor = int(rng.choice(donors))
        centroids[k] = centroids[donor] + rng.normal(scale=eps, size=centroids.shape[1])
        donor_pts = np.where(labels == donor)[0]
        rng.shuffle(donor_pts)
        half = donor_pts[: max(1, donor_pts.size // 2)]
        labels[half] = k
        counts = np.bincount(labels, minlength=num_clusters)

    return labels
