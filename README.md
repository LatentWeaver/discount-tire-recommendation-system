# HGT Two-Tower Retrieval — MovieLens-100K

A two-tower retrieval recommender built on a **Heterogeneous Graph Transformer
(HGT)** encoder, trained on the **MovieLens-100K** dataset (50 000-rating
random subsample).

> **Branch:** `hgt-two-tower-movielen`

---

## Architecture

```
  Heterogeneous Graph
  ───────────────────
   ┌──────┐  reviews   ┌──────┐  belongs_to   ┌────────┐
   │ user ├───rating──▶│ item ├──────────────▶│ genre  │
   └──────┘            │      │               └────────┘
                       │      │  has_spec     ┌────────┐
                       │      ├──────────────▶│ decade │
                       └──────┘               └────────┘

  HGT encoder ──▶ h_user, h_item, h_genre, h_decade
       │
       ├─▶ ItemTower([h_item ⊕ h_genre ⊕ h_decade ⊕ item_features])  ──▶ item_vec
       │
       └─▶ UserTower([h_user ⊕ mean(item_vec over train history)])    ──▶ user_vec

  score(u, t) = user_vec · item_vec / temperature       (ℓ2-normalised → cosine)
```

The node-type slots are kept as **`user` / `tire` / `brand` / `size`** in the
graph schema so the encoder, sampler, trainer, and evaluator stay
domain-agnostic. For MovieLens:

| Slot    | Maps to                                       |
|---------|-----------------------------------------------|
| `user`  | MovieLens user_id (943 users)                 |
| `tire`  | Movie / item (1 586 movies)                   |
| `brand` | Primary genre (first listed of 19 genres)     |
| `size`  | Release decade (`1920s` ... `2000s`, 9 buckets) |

Item features (23 dims): scaled `[avg_rating, rating_std, rating_count, year]`
+ 19 binary genre flags.

---

## Quick start

```bash
# 1. Download MovieLens-100K and write the 50 000-rating subsample.
uv run python scripts/subsample_movielens.py

# 2. Build the heterogeneous graph from the subsample.
uv run python scripts/build_graph_movielens.py

# 3. Train. Default --graph-path points at the MovieLens .pt.
uv run python scripts/train.py --epochs 30

# 4. (Optional) Build a FAISS index from the trained checkpoint.
uv run python scripts/build_index.py \
    --checkpoint outputs/checkpoints/two_tower_e30.pt
```

Useful flags on `train.py`:

- `--loss {softmax|bpr|bce}` — retrieval loss (default `softmax`).
- `--hard-neg-k N` — hard negatives per positive, drawn from (brand ∪ size).
- `--ssl-lambda 0.1` — SGL-style graph contrastive auxiliary loss.
- `--history-drop 0.3` — drop a fraction of train positives before history pooling (regularisation).
- `--device cpu|mps|cuda` — override auto-detection.

---

## Data

| File                              | Contents                                     |
|-----------------------------------|----------------------------------------------|
| `data/raw/ml-100k/u.data`         | Full 100 K ratings (TSV)                     |
| `data/raw/ml-100k/u.data.50k`     | 50 K random sample (seed 42)                 |
| `data/raw/ml-100k/u.user.50k`     | Users in sample (943)                        |
| `data/raw/ml-100k/u.item.50k`     | Movies in sample (1 586) + 19 genre flags    |
| `data/processed/hetero_graph_movielens.pt` | Built `HeteroData` payload          |

`data/processed/hetero_graph_movielens.pt` carries the graph, the
string↔index mappings, the per-movie metadata DataFrame, and the per-review
DataFrame (with the static feature block in `df.attrs` so the sampler can
recompute train-only rating aggregates without losing the static columns).

---

## Source layout

```
configs/movielens.yaml              # paths + graph-build switches
scripts/
  subsample_movielens.py            # download + 50K random subsample
  build_graph_movielens.py          # raw data → HeteroData .pt
  train.py                          # main training entry
  evaluate.py                       # load a checkpoint, recompute metrics
  build_index.py                    # FAISS index from a checkpoint
  visualize_graph.py                # schema/degree/feature plots
src/
  data_processing/
    graph_builder.py                # HeteroData construction (generic)
    preprocessing_movielens.py      # MovieLens → DataFrames + features
  models/
    hgt_layer.py                    # HGT message-passing layer
    hgt_encoder.py                  # stacked HGT + input projections
    two_tower.py                    # ItemTower / UserTower / score
  training/
    sampler.py                      # pair-level train/val/test split, BPR sampler
    trainer.py                      # one HGT forward per step + chosen loss
    augment.py                      # SGL-style edge/feature dropout
    hard_negatives.py               # brand ∪ size bucket-based negatives
    evaluation.py                   # Recall/NDCG/HitRate@K with train-mask
tests/
  test_hgt_forward.py               # encoder smoke test
  test_train_step.py                # end-to-end pipeline smoke test
  test_split_integrity.py           # split / mask leak checks
```

---

## Splits & leakage protection

- **Pair-level split.** The atomic split unit is the `(user, item)` pair,
  not the individual review edge, so duplicate reviews of the same item
  cannot leak across train / val / test.
- **Tire-feature leak fix.** After the split, the sampler overwrites
  `data["tire"].x` with per-movie rating aggregates recomputed from
  **train edges only**. Static features (release year, genre flags) carry
  through unchanged.
- **Train-mask in eval.** During Recall@K evaluation, all items the user
  reviewed in the train split are masked out before top-K is computed
  (except the held-out positive itself).
- **Train-only message passing.** The encoder only ever sees the
  train-only review graph; held-out edges never participate in message
  passing.

---

## Loss

Default is sampled-softmax over in-batch negatives with a learnable
temperature. `bpr` (pairwise BPR with one random negative) and `bce`
(binary cross-entropy with one random negative) are also available via
`--loss`. Hard negatives are drawn from the union of an item's brand and
size buckets so the model has to discriminate among substitutes, not just
random catalog entries.

---

## Smoke-test reference

`uv run python scripts/train.py --epochs 2 --steps-per-epoch 50 --batch-size 256 --device cpu`

```
Train: 22 016 | Val: 2 773 | Test: 2 766
[epoch 1] loss=5.6252  Recall@10=0.0137  Recall@20=0.0260  Recall@50=0.0685
[epoch 2] loss=5.5082  Recall@10=0.0083  Recall@20=0.0184  Recall@50=0.0548
test  Recall@20=0.0296  NDCG@20=0.0093  HitRate@50=0.0770
```

This is the 100-step warm-up baseline — bumping `--epochs 30` and
`--steps-per-epoch 200` (the script defaults) is the actual training run.
