"""
PyG LinkNeighborLoader wrapper for CUDA-friendly mini-batch HGT training.

This path uses PyG's compiled neighbor sampling backend when ``pyg-lib`` or
``torch-sparse`` is installed. It is intended for remote CUDA machines; local
MPS environments can keep using the pure-Python ``HGTSubgraphSampler``.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from src.training.sampler import BPRSampler


def has_compiled_neighbor_sampler() -> bool:
    try:
        import pyg_lib  # noqa: F401

        return True
    except ModuleNotFoundError:
        pass

    try:
        import torch_sparse  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


class PyGLinkNeighborBatcher:
    def __init__(
        self,
        sampler: BPRSampler,
        batch_size: int,
        num_hops: int = 2,
        fanout: int = 8,
        neg_sampling_ratio: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        if not has_compiled_neighbor_sampler():
            raise RuntimeError(
                "PyG LinkNeighborLoader requires pyg-lib or torch-sparse. "
                "Install pyg-lib on CUDA, or use --sampler-backend python."
            )

        self.sampler = sampler
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.fanout = fanout
        self.neg_sampling_ratio = neg_sampling_ratio
        self.num_workers = num_workers

        data = sampler.train_data.cpu()
        edge_label_index = torch.stack(
            [sampler.train_users.cpu(), sampler.train_items.cpu()]
        )
        edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float32)
        num_neighbors = {
            edge_type: [fanout for _ in range(num_hops)] for edge_type in data.edge_types
        }

        self.loader = LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            edge_label_index=(("user", "reviews", "item"), edge_label_index),
            edge_label=edge_label,
            neg_sampling_ratio=neg_sampling_ratio,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            transform=self._attach_global_node_ids,
        )
        self._iterator: Iterator[HeteroData] = iter(self.loader)

    def sample(self) -> HeteroData:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)

    @staticmethod
    def _attach_global_node_ids(batch: HeteroData) -> HeteroData:
        for node_type in batch.node_types:
            n_id = getattr(batch[node_type], "n_id", None)
            if n_id is not None:
                batch[node_type].node_id = n_id
        return batch
