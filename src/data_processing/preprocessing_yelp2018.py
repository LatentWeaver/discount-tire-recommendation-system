"""
Preprocessing pipeline for the LightGCN Yelp 2018 dataset.

Maps the LightGCN release at https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018
onto the same node-type schema used by ``graph_builder``:

    user  slot  ← LightGCN user_id (remapped int)
    tire  slot  ← LightGCN business item_id (remapped int)
    brand slot  ← single "ALL" node (no item categories in this release)
    size  slot  ← single "ALL" node (no item attributes in this release)

The slot names ("tire", "brand", "size") are kept so the encoder, sampler,
trainer, and evaluator continue to work unchanged — they treat the slots
as opaque node types.

Inputs (under ``data/raw/yelp2018/``):
    user_list.txt   header + rows: org_id remap_id
    item_list.txt   header + rows: org_id remap_id
    train.txt       per line: user_id item1 item2 ...
    test.txt        per line: user_id item1 item2 ...

The train / test files already encode LightGCN's official split; we
preserve it verbatim so our reported metrics are directly comparable to
the LightGCN paper.

Implicit feedback only — every (user, item) line is a positive
interaction. No ratings, no timestamps, no item content features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Yelp2018Interactions:
    """LightGCN Yelp 2018 dataset materialised as flat edge arrays.

    Both ``train_edges`` and ``test_edges`` are (N, 2) int64 arrays with
    columns [user_remap_id, item_remap_id]. Counts come from the canonical
    user_list / item_list files (so the ID space stays dense and matches
    LightGCN's index layout exactly).
    """

    num_users: int
    num_items: int
    train_edges: np.ndarray  # (n_train, 2)
    test_edges: np.ndarray   # (n_test, 2)


def _load_id_list(path: Path) -> int:
    """Return count of remapped ids in a LightGCN id-list file.

    Header line is ``org_id remap_id``; remaining rows are one mapping each.
    """
    with open(path) as f:
        header = f.readline()
        if not header.strip().startswith("org_id"):
            raise ValueError(
                f"{path}: expected first line to start with 'org_id', got {header!r}"
            )
        return sum(1 for _ in f)


def _load_user_item_file(path: Path) -> np.ndarray:
    """Parse LightGCN ``train.txt`` / ``test.txt`` into a (N, 2) int64 array.

    Each line is ``user_id item1 item2 ...``. Lines with no items are skipped.
    """
    src: list[int] = []
    dst: list[int] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for tok in parts[1:]:
                src.append(u)
                dst.append(int(tok))
    if not src:
        raise ValueError(f"{path}: no interactions parsed.")
    return np.stack([np.asarray(src, dtype=np.int64),
                     np.asarray(dst, dtype=np.int64)], axis=1)


def load_lightgcn_yelp2018(raw_dir: str | Path) -> Yelp2018Interactions:
    """Load all four LightGCN Yelp 2018 files from ``raw_dir``."""
    raw = Path(raw_dir)
    num_users = _load_id_list(raw / "user_list.txt")
    num_items = _load_id_list(raw / "item_list.txt")
    train_edges = _load_user_item_file(raw / "train.txt")
    test_edges = _load_user_item_file(raw / "test.txt")

    if int(train_edges[:, 0].max()) >= num_users or int(test_edges[:, 0].max()) >= num_users:
        raise ValueError("User id out of range relative to user_list.txt.")
    if int(train_edges[:, 1].max()) >= num_items or int(test_edges[:, 1].max()) >= num_items:
        raise ValueError("Item id out of range relative to item_list.txt.")

    return Yelp2018Interactions(
        num_users=num_users,
        num_items=num_items,
        train_edges=train_edges,
        test_edges=test_edges,
    )


def carve_val_from_train(
    train_edges: np.ndarray,
    val_ratio: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly hold out ``val_ratio`` of train edges as a validation slice.

    LightGCN itself does not publish a validation split; we carve one so
    the trainer has an early-stopping signal without ever touching the
    paper's test split. The test edges remain LightGCN's verbatim 324K.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")
    if val_ratio == 0.0:
        return train_edges, np.empty((0, 2), dtype=train_edges.dtype)
    rng = np.random.default_rng(seed)
    n = train_edges.shape[0]
    perm = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_edges[train_idx], train_edges[val_idx]
