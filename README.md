# HGT Two-Tower Retrieval — LightGCN Yelp 2018

A two-tower retrieval recommender built on a **Heterogeneous Graph Transformer
(HGT)** encoder, trained on the **LightGCN Yelp 2018** benchmark dataset
([source](https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018)) so
results are directly comparable to the LightGCN paper.

> **Branch:** `two-tower-yelp2018`

---

## Architecture

```
  Heterogeneous Graph
  ───────────────────
   ┌──────┐  reviews    ┌──────┐  belongs_to   ┌────────┐
   │ user ├────────────▶│ item ├──────────────▶│ brand* │
   └──────┘             │      │               └────────┘
                        │      │  has_spec     ┌────────┐
                        │      ├──────────────▶│ size*  │
                        └──────┘               └────────┘

  *Yelp 2018 has no item categories or attributes — brand and size
   collapse to a single "ALL" node each so the HGT slots stay populated.

  HGT encoder ──▶ h_user, h_item, h_brand, h_size
       │
       ├─▶ ItemTower([h_item ⊕ h_brand ⊕ h_size ⊕ item_features])  ──▶ item_vec
       │
       └─▶ UserTower([h_user ⊕ mean(item_vec over train history)])  ──▶ user_vec

  score(u, t) = user_vec · item_vec / temperature       (ℓ2-normalised → cosine)
```

The node-type slots are kept as **`user` / `tire` / `brand` / `size`** in the
graph schema so the encoder, sampler, trainer, and evaluator stay
domain-agnostic. For Yelp 2018:

| Slot    | Maps to                                                    |
|---------|------------------------------------------------------------|
| `user`  | LightGCN user_id (31 668 users)                            |
| `tire`  | LightGCN business item_id (38 048 items)                   |
| `brand` | Single "ALL" node — LightGCN release ships no categories   |
| `size`  | Single "ALL" node — LightGCN release ships no attributes   |

Item features: a single zero placeholder (`tire.x = zeros(n_items, 1)`) since
LightGCN provides ID-only items. The HGT learns purely from interaction
structure, matching LightGCN's setup.

---

## Quick start

```bash
# 1. Build the heterogeneous graph from the LightGCN raw files.
#    Expects data/raw/yelp2018/{train,test,user_list,item_list}.txt
uv run python scripts/build_graph_yelp2018.py

# 2. Train. Default --graph-path points at the Yelp 2018 .pt.
uv run python scripts/train.py --epochs 30

# 3. (Optional) Build a FAISS index from the trained checkpoint.
uv run python scripts/build_index.py \
    --checkpoint outputs/checkpoints/two_tower_e30.pt
```

Useful flags on `train.py`:

- `--loss {softmax|bpr|bce}` — retrieval loss (default `softmax`).
- `--hard-neg-k N` — hard negatives per positive, drawn from (brand ∪ size)
  buckets. With single-bucket brand/size on Yelp this degenerates to random
  negatives; keep at 0 unless brand/size carry signal.
- `--ssl-lambda 0.1` — SGL-style graph contrastive auxiliary loss.
- `--history-drop 0.3` — drop a fraction of train positives before history pooling (regularisation).
- `--device cpu|mps|cuda` — override auto-detection.

For a fast smoke test, build a 10% subsample first:

```bash
uv run python scripts/build_graph_yelp2018.py --subsample-ratio 0.1
uv run python scripts/train.py \
    --graph-path data/processed/hetero_graph_yelp2018_sub10.pt \
    --epochs 1 --steps-per-epoch 10 --batch-size 256
```

---

## Data

| File                                       | Contents                                |
|--------------------------------------------|-----------------------------------------|
| `data/raw/yelp2018/user_list.txt`          | 31 668 user remap-id mappings           |
| `data/raw/yelp2018/item_list.txt`          | 38 048 business remap-id mappings       |
| `data/raw/yelp2018/train.txt`              | 1 237 259 train interactions            |
| `data/raw/yelp2018/test.txt`               | 324 147 test interactions               |
| `data/processed/hetero_graph_yelp2018.pt`  | Built `HeteroData` payload              |

The payload carries the graph and a `precomputed_split` dict
(`train_idx`, `val_idx`, `test_idx`) recording the row ranges in
`(user, reviews, tire).edge_index` that correspond to each partition.
LightGCN's test split is preserved verbatim; a 5% slice is carved out
of the train edges for early-stopping (validation), so the test partition
is never touched during training.

---

## Source layout

```
configs/yelp2018.yaml                # paths + graph-build switches
scripts/
  build_graph_yelp2018.py            # LightGCN txt → HeteroData .pt
  train.py                           # main training entry
  evaluate.py                        # load a checkpoint, recompute metrics
  build_index.py                     # FAISS index from a checkpoint
src/
  data_processing/
    graph_builder.py                 # HeteroData construction (generic)
    preprocessing_yelp2018.py        # LightGCN files → flat edge arrays
  models/
    hgt_layer.py                     # HGT message-passing layer
    hgt_encoder.py                   # stacked HGT + input projections
    two_tower.py                     # ItemTower / UserTower / score
  training/
    sampler.py                       # honors precomputed split, BPR + implicit
    trainer.py                       # one HGT forward per step + chosen loss
    augment.py                       # SGL-style edge/feature dropout
    hard_negatives.py                # brand ∪ size bucket-based negatives
    evaluation.py                    # Recall/NDCG/HitRate@K with train-mask
```

---

## Splits & leakage protection

- **LightGCN's split is honored verbatim.** The sampler reads the
  `precomputed_split` from the payload and skips its random splitter, so
  metrics are directly comparable to the LightGCN paper.
- **Train-mask in eval.** During Recall@K evaluation, all items the user
  interacted with in the train split are masked out before top-K is computed
  (except the held-out positive itself).
- **Train-only message passing.** The encoder only ever sees the
  train-only interaction graph; val and test edges never participate in
  message passing.

---

## Loss

Default is sampled-softmax over in-batch negatives with a learnable
temperature. `bpr` (pairwise BPR with one random negative) and `bce`
(binary cross-entropy with one random negative) are also available via
`--loss`. Yelp 2018 is implicit-feedback (every interaction is a positive);
the rating-threshold contrastive path is disabled automatically when the
graph has no `edge_attr`.
