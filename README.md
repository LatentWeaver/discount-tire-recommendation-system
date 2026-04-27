# HGT Recommendation Benchmark

Heterogeneous Graph Transformer recommendation benchmark focused on LastFM.

The active graph schema is:

```text
user  --reviews--> item(artist)
item  --rev_by-->  user
user  --follows--> user
item  --has_tag--> tag
tag   --tag_of-->  item
```

Training uses mini-batch heterogeneous subgraph sampling. On CUDA, install
`pyg-lib` to enable PyG's compiled `LinkNeighborLoader`; otherwise
`--sampler-backend auto` falls back to the portable Python sampler.

## Setup

```bash
uv sync
```

## Build LastFM Graph

```bash
uv run python scripts/build_lastfm_graph.py
```

This creates:

```text
data/processed/lastfm_hetero_graph.pt
```

## Train HGT

```bash
uv run python scripts/pretrain_hgt.py \
  --graph-path data/processed/lastfm_hetero_graph.pt \
  --epochs 20 \
  --steps-per-epoch 200 \
  --training-mode subgraph \
  --sampler-backend auto \
  --batch-size 1024 \
  --hidden-dim 64 \
  --fanout 8 \
  --save-path outputs/checkpoints/hgt_lastfm.pt
```

Use `--sampler-backend pyg-link --device cuda` on a CUDA machine with `pyg-lib`
installed. LastFM is implicit feedback, so focus on `Recall@K`, `NDCG@K`, and
`HitRate@K`; review classification metrics are not meaningful for this graph.

## Smoke Test

```bash
uv run python tests/test_hgt_recommender.py
```

## Quick Evaluation

```bash
uv run python scripts/evaluate_hgt.py \
  --checkpoint outputs/checkpoints/hgt_lastfm.pt \
  --graph-path data/processed/lastfm_hetero_graph.pt \
  --split val \
  --max-eval-examples 1000 \
  --sample-eval-examples
```

Remove `--max-eval-examples` when you want the full validation/test set.
