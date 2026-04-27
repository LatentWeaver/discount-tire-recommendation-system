"""
Mini-batch heterogeneous subgraph sampler for HGT training.

The sampler follows the HGT paper's practical training idea: start from a
typed batch of target nodes and expand a local heterogeneous neighborhood
across meta-relations, with a fanout cap per relation and hop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

from src.training.sampler import BPRSampler


EdgeType = tuple[str, str, str]


@dataclass
class SubgraphBatch:
    data: HeteroData
    users: torch.Tensor
    pos_items: torch.Tensor
    neg_items: torch.Tensor
    observed_users: torch.Tensor
    observed_items: torch.Tensor
    observed_labels: torch.Tensor


@dataclass
class NodeSubgraphBatch:
    data: HeteroData
    node_type: str
    global_node_ids: torch.Tensor
    local_node_ids: torch.Tensor


class HGTSubgraphSampler:
    def __init__(
        self,
        sampler: BPRSampler,
        num_hops: int = 2,
        fanout: int = 20,
        max_nodes_per_type: int = 4096,
        seed: int = 0,
    ) -> None:
        self.sampler = sampler
        self.data = sampler.train_data.cpu()
        self.num_hops = num_hops
        self.fanout = fanout
        self.max_nodes_per_type = max_nodes_per_type
        self._gen = torch.Generator().manual_seed(seed)
        self._adj = self._build_adjacency(self.data)

    def sample(self, batch_size: int) -> SubgraphBatch:
        users, pos_items, neg_items = self.sampler.sample(batch_size)
        obs_users, obs_items, obs_labels = self.sampler.sample_observed(batch_size)

        node_sets = {
            nt: set[int]() for nt in self.data.node_types
        }
        node_sets["user"].update(users.tolist())
        node_sets["user"].update(obs_users.tolist())
        node_sets["item"].update(pos_items.tolist())
        node_sets["item"].update(neg_items.tolist())
        node_sets["item"].update(obs_items.tolist())

        sampled_edges = self._expand(node_sets)
        subgraph, local_maps = self._induce(node_sets, sampled_edges)

        return SubgraphBatch(
            data=subgraph,
            users=self._to_local(users, local_maps["user"]),
            pos_items=self._to_local(pos_items, local_maps["item"]),
            neg_items=self._to_local(neg_items, local_maps["item"]),
            observed_users=self._to_local(obs_users, local_maps["user"]),
            observed_items=self._to_local(obs_items, local_maps["item"]),
            observed_labels=obs_labels,
        )

    def sample_nodes(
        self,
        node_type: str,
        node_ids: torch.Tensor,
    ) -> NodeSubgraphBatch:
        node_sets = {nt: set[int]() for nt in self.data.node_types}
        node_sets[node_type].update(node_ids.cpu().tolist())

        sampled_edges = self._expand(node_sets)
        subgraph, local_maps = self._induce(node_sets, sampled_edges)

        return NodeSubgraphBatch(
            data=subgraph,
            node_type=node_type,
            global_node_ids=node_ids.cpu(),
            local_node_ids=self._to_local(node_ids.cpu(), local_maps[node_type]),
        )

    def _expand(self, node_sets: dict[str, set[int]]) -> dict[EdgeType, set[int]]:
        sampled_edges = {edge_type: set[int]() for edge_type in self.data.edge_types}
        frontier = {nt: set(ids) for nt, ids in node_sets.items()}
        for _ in range(self.num_hops):
            next_frontier = {nt: set() for nt in self.data.node_types}
            for edge_type in self.data.edge_types:
                src_type, _, dst_type = edge_type
                adj = self._adj[edge_type]
                for src in frontier[src_type]:
                    for dst, edge_id in self._sample_edges(adj["src"].get(src, [])):
                        sampled_edges[edge_type].add(edge_id)
                        if self._can_add_node(node_sets, dst_type, dst):
                            node_sets[dst_type].add(dst)
                            next_frontier[dst_type].add(dst)
                for dst in frontier[dst_type]:
                    for src, edge_id in self._sample_edges(adj["dst"].get(dst, [])):
                        sampled_edges[edge_type].add(edge_id)
                        if self._can_add_node(node_sets, src_type, src):
                            node_sets[src_type].add(src)
                            next_frontier[src_type].add(src)
            frontier = next_frontier
        return sampled_edges

    def _can_add_node(
        self,
        node_sets: dict[str, set[int]],
        node_type: str,
        node_id: int,
    ) -> bool:
        if node_id in node_sets[node_type]:
            return False
        return self.max_nodes_per_type <= 0 or (
            len(node_sets[node_type]) < self.max_nodes_per_type
        )

    def _induce(
        self,
        node_sets: dict[str, set[int]],
        sampled_edges: dict[EdgeType, set[int]],
    ) -> tuple[HeteroData, dict[str, dict[int, int]]]:
        subgraph = HeteroData()
        local_maps: dict[str, dict[int, int]] = {}
        global_ids: dict[str, torch.Tensor] = {}

        for nt in self.data.node_types:
            ids = sorted(node_sets[nt])
            ids_t = torch.tensor(ids, dtype=torch.long)
            global_ids[nt] = ids_t
            local_maps[nt] = {node_id: i for i, node_id in enumerate(ids)}
            subgraph[nt].num_nodes = len(ids)
            subgraph[nt].node_id = ids_t
            x = getattr(self.data[nt], "x", None)
            if x is not None:
                subgraph[nt].x = x.index_select(0, ids_t)

        for edge_type in self.data.edge_types:
            src_type, _, dst_type = edge_type
            store = self.data[edge_type]
            src_map = local_maps[src_type]
            dst_map = local_maps[dst_type]
            kept_edge_ids: list[int] = []
            local_src_list: list[int] = []
            local_dst_list: list[int] = []
            edge_ids = sorted(sampled_edges[edge_type])
            if edge_ids:
                edge_ids_t = torch.tensor(edge_ids, dtype=torch.long)
                kept_global = store.edge_index.index_select(1, edge_ids_t)
                for edge_id, (src, dst) in zip(edge_ids, kept_global.t().tolist()):
                    if src in src_map and dst in dst_map:
                        kept_edge_ids.append(edge_id)
                        local_src_list.append(src_map[src])
                        local_dst_list.append(dst_map[dst])

            keep_idx = torch.tensor(kept_edge_ids, dtype=torch.long)
            local_src = torch.tensor(local_src_list, dtype=torch.long)
            local_dst = torch.tensor(local_dst_list, dtype=torch.long)
            subgraph[edge_type].edge_index = torch.stack([local_src, local_dst])

            edge_attr = getattr(store, "edge_attr", None)
            if edge_attr is not None:
                subgraph[edge_type].edge_attr = edge_attr.index_select(0, keep_idx)
            review_edge_id = getattr(store, "review_edge_id", None)
            if review_edge_id is not None:
                subgraph[edge_type].review_edge_id = review_edge_id.index_select(
                    0, keep_idx
                )

        return subgraph, local_maps

    def _sample_edges(self, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if self.fanout <= 0 or len(edges) <= self.fanout:
            return edges
        idx = torch.randperm(len(edges), generator=self._gen)[: self.fanout].tolist()
        return [edges[i] for i in idx]

    @staticmethod
    def _to_local(values: torch.Tensor, local_map: dict[int, int]) -> torch.Tensor:
        return torch.tensor([local_map[int(v)] for v in values.tolist()], dtype=torch.long)

    @staticmethod
    def _build_adjacency(
        data: HeteroData,
    ) -> dict[EdgeType, dict[str, dict[int, list[tuple[int, int]]]]]:
        out: dict[EdgeType, dict[str, dict[int, list[tuple[int, int]]]]] = {}
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index.cpu()
            by_src: dict[int, list[tuple[int, int]]] = {}
            by_dst: dict[int, list[tuple[int, int]]] = {}
            for edge_id, (src, dst) in enumerate(edge_index.t().tolist()):
                by_src.setdefault(src, []).append((dst, edge_id))
                by_dst.setdefault(dst, []).append((src, edge_id))
            out[edge_type] = {"src": by_src, "dst": by_dst}
        return out
