#!/usr/bin/env python3
"""
Data-leakage audit for the vehicle-as-query graph + BPRSampler split.

Checks:
  1. train ∩ val == ∅,  train ∩ test == ∅,  val ∩ test == ∅
  2. train_idx + val_idx + test_idx covers all review edges exactly once
  3. message-passing graph (train_data) only contains train edges (forward + reverse)
  4. eval mask (user_reviewed_train) is built from train edges only
  5. tire features (data["tire"].x) — do per-tire aggregates leak val/test ratings?
  6. user/tire id used for evaluation actually appear in train (no cold-start contamination)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sampler import BPRSampler


def main() -> None:
    payload = torch.load(PROJECT_ROOT / "data/processed/hetero_graph_vehicle.pt",
                         weights_only=False)
    data = payload["graph"]
    review_df = payload.get("review_df")
    tire_df = payload.get("tire_df")

    print("=" * 60)
    print(" Data leakage audit — vehicle graph")
    print("=" * 60)

    sampler = BPRSampler(
        data,
        rating_threshold=4.0,
        seed=0,
        review_df=payload.get("review_df"),
    )
    split = sampler.split

    n_total = data["user", "reviews", "tire"].edge_index.size(1)
    s_train = set(split.train_idx.tolist())
    s_val = set(split.val_idx.tolist())
    s_test = set(split.test_idx.tolist())

    # ─── 1. Disjoint splits ─────────────────────────────────────────
    print("\n[1] Split disjointness")
    print(f"   |train|={len(s_train)}  |val|={len(s_val)}  |test|={len(s_test)}  total={n_total}")
    print(f"   train ∩ val   = {len(s_train & s_val)}")
    print(f"   train ∩ test  = {len(s_train & s_test)}")
    print(f"   val   ∩ test  = {len(s_val & s_test)}")
    union = s_train | s_val | s_test
    print(f"   union size    = {len(union)}  (expect {n_total})")
    print(f"   missing       = {len(set(range(n_total)) - union)}")

    # ─── 2. Train-only message-passing graph ────────────────────────
    print("\n[2] Message-passing graph (sampler.train_data)")
    fw = ("user", "reviews", "tire")
    rv = ("tire", "rev_by", "user")
    ei_fw = sampler.train_data[fw].edge_index
    ei_rv = sampler.train_data[rv].edge_index
    print(f"   train_data forward edges:  {ei_fw.size(1)}  (expect {len(s_train)})")
    print(f"   train_data reverse edges:  {ei_rv.size(1)}  (expect {len(s_train)})")

    # check no val/test edge appears in train_data
    full_fw = data[fw].edge_index
    train_fw_pairs = {(int(u), int(t)) for u, t in zip(ei_fw[0], ei_fw[1])}
    val_pairs = {(int(full_fw[0, i]), int(full_fw[1, i])) for i in s_val}
    test_pairs = {(int(full_fw[0, i]), int(full_fw[1, i])) for i in s_test}
    leak_val = train_fw_pairs & val_pairs
    leak_test = train_fw_pairs & test_pairs
    print(f"   val pairs leaking into train graph:  {len(leak_val)}  ⚠️" if leak_val else
          "   val pairs leaking into train graph:  0  ✓")
    print(f"   test pairs leaking into train graph: {len(leak_test)}  ⚠️" if leak_test else
          "   test pairs leaking into train graph: 0  ✓")

    # NB: a (u, t) pair may legitimately appear in BOTH train and val/test if a
    # vehicle reviewed the same tire twice. Distinguish by edge index, not pair.

    # ─── 3. Eval mask built from train only ─────────────────────────
    print("\n[3] Eval-mask composition (sampler.user_reviewed_train)")
    mask_pairs = {
        (u, t)
        for u, seen in enumerate(sampler.user_reviewed_train)
        for t in seen
    }
    train_pairs_full = {
        (int(full_fw[0, i]), int(full_fw[1, i])) for i in s_train
    }
    val_only_pairs = val_pairs - train_pairs_full
    test_only_pairs = test_pairs - train_pairs_full
    leaked_val = mask_pairs & val_only_pairs
    leaked_test = mask_pairs & test_only_pairs
    print(f"   val-only pairs in eval-mask:  {len(leaked_val)}  ⚠️" if leaked_val else
          "   val-only pairs in eval-mask:  0  ✓")
    print(f"   test-only pairs in eval-mask: {len(leaked_test)}  ⚠️" if leaked_test else
          "   test-only pairs in eval-mask: 0  ✓")

    # ─── 4. Are eval users/tires reachable from train? ──────────────
    print("\n[4] Eval coverage (cold-start contamination check)")
    train_users_set = set(int(u) for u, _ in train_pairs_full)
    train_tires_set = set(int(t) for _, t in train_pairs_full)
    val_users = set(sampler.val_users.tolist())
    val_tires = set(sampler.val_tires.tolist())
    test_users = set(sampler.test_users.tolist())
    test_tires = set(sampler.test_tires.tolist())
    print(f"   val users not in train:   {len(val_users - train_users_set)} / {len(val_users)}")
    print(f"   val tires not in train:   {len(val_tires - train_tires_set)} / {len(val_tires)}")
    print(f"   test users not in train:  {len(test_users - train_users_set)} / {len(test_users)}")
    print(f"   test tires not in train:  {len(test_tires - train_tires_set)} / {len(test_tires)}")

    # ─── 5. Tire feature leakage (THE BIG QUESTION) ─────────────────
    print("\n[5] Tire feature leakage  (data['tire'].x)")
    if review_df is not None:
        # Compare graph['tire'].x (built over ALL reviews) vs
        # sampler.train_data['tire'].x (recomputed over train rows by sampler).
        x_full = data["tire"].x
        x_train = sampler.train_data["tire"].x
        if x_full.shape == x_train.shape:
            same = torch.allclose(x_full, x_train)
            diff = (x_full - x_train).abs()
            print(f"    full-aggregate vs train-only feature matrix:")
            print(f"      identical: {same}")
            print(f"      max abs diff: {diff.max().item():.4f}")
            print(f"      mean abs diff: {diff.mean().item():.4f}")
            print(f"      rows that changed: {int((diff.sum(dim=1) > 1e-6).sum().item())}/{x_full.size(0)}")
            if not same:
                print("    → sampler.train_data['tire'].x is the LEAK-FREE version")
                print("      used by the trainer. data['tire'].x (the bundled graph)")
                print("      still holds the full-aggregate version — but trainer")
                print("      only consumes train_data, so the leak is closed.  ✓")
        else:
            print(f"    shape mismatch: full {tuple(x_full.shape)} vs"
                  f" train {tuple(x_train.shape)}  ⚠️")
    else:
        print("    (sampler called without review_df — feature leak NOT closed)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
