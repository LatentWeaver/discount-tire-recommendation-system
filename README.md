# HGT Recommendation Benchmark

Heterogeneous Graph Transformer recommendation benchmark using MovieLens.

The active graph schema is:

```text
user  --reviews-->  item
item  --rev_by-->   user
item  --has_genre--> genre
genre --genre_of--> item
```

Training combines BPR ranking with an observed-review liked/disliked
classification head.

## Setup

```bash
uv sync
```

## Build MovieLens Graph

```bash
uv run python scripts/build_movielens_graph.py
```

This creates:

```text
data/processed/movielens_hetero_graph.pt
```

## Pretrain HGT

```bash
uv run python scripts/pretrain_hgt.py \
  --epochs 20 \
  --steps-per-epoch 20 \
  --batch-size 1024 \
  --save-path outputs/checkpoints/hgt_movielens_pretrained.pt
```

Metrics reported:

- `Recall@K`, `NDCG@K`, `HitRate@K` for top-K ranking.
- `ReviewBalancedAcc`, `ReviewPosAcc`, `ReviewNegAcc` for liked/disliked prediction.

## Smoke Test

```bash
uv run python tests/test_hgt_recommender.py
```
