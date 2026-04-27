# HGT Recommendation Benchmark

Heterogeneous Graph Transformer recommendation benchmark using MovieLens.

The active graph schema is:

```text
user  --reviews-->  item
item  --rev_by-->   user
user  --rates_low/mid/high--> item
item  --rated_low/mid/high_by--> user
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

To train on MovieLens 1M instead:

```bash
uv run python scripts/build_movielens_graph.py --dataset 1m
```

This creates:

```text
data/processed/movielens_1m_hetero_graph.pt
```

## Pretrain HGT

```bash
uv run python scripts/pretrain_hgt.py \
  --graph-path data/processed/movielens_1m_hetero_graph.pt \
  --epochs 20 \
  --steps-per-epoch 200 \
  --training-mode subgraph \
  --batch-size 256 \
  --fanout 8 \
  --max-nodes-per-type 4096 \
  --save-path outputs/checkpoints/hgt_movielens_pretrained.pt
```

`subgraph` training samples a typed multi-hop heterogeneous neighborhood for
each interaction batch, so MovieLens 1M does not require full-graph HGT
encoding during training. Set `--eval-every N` only when you explicitly want
full-graph validation; the default `0` skips it to avoid MPS memory pressure.

Metrics reported:

- `Recall@K`, `NDCG@K`, `HitRate@K` for top-K ranking.
- `ReviewBalancedAcc`, `ReviewPosAcc`, `ReviewNegAcc` for liked/disliked prediction.

## Smoke Test

```bash
uv run python tests/test_hgt_recommender.py
```

## Quick Evaluation

```bash
uv run python scripts/evaluate_hgt.py \
  --checkpoint outputs/checkpoints/hgt_movielens_pretrained.pt \
  --split val \
  --max-eval-examples 1000 \
  --sample-eval-examples
```

Remove `--max-eval-examples` when you want the full validation/test set.
