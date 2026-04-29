# GNN-based Tire Recommendation System

A two-tower retrieval recommender for tires built on a **Heterogeneous Graph Transformer (HGT)** encoder. The system maps **vehicle models (year-make-model)** to tires using per-vehicle aggregated review signals — explicitly trading per-user personalization for the much higher density of vehicle-level interactions.

> **Branch status (`two-tower-hgt`):** vehicle-as-query node, leak-free split, review-text fusion. Final metric on the held-out test split: **Recall@20 = 0.68**, **Recall@50 = 0.94**.

---

## Why vehicle-as-query

A typical tire-review dataset is brutally sparse on the user side:

| Entity | Distinct | Mean reviews/entity |
|---|---|---|
| user (author) | 886 | 1.24 |
| **(make, model)** | **277** | **3.88** |
| (make, model, year) | 755 | 1.42 |

Customers buy tires once every few years, so per-user signals barely exist. A *vehicle model*, by contrast, has dozens of owners writing about the same fitment / load class / tread expectations — which is exactly the dimension that determines tire suitability. Aggregating to **(make, model)** gives ~3× the density of user-level data while keeping the recommendation question physically grounded ("what tires work well for an F-150?").

Going finer than `(make, model)` (e.g. adding `year` or `trim`) collapses density below the user baseline, so we stop at model.

---

## Heterogeneous Graph Schema

```
   ┌─────────┐    reviews     ┌────────┐     belongs_to    ┌────────┐
   │ vehicle ├───(rating)────▶│  tire  │──────────────────▶│ brand  │
   │ (m, m)  │                │        │                    └────────┘
   │  277    │                │   63   │     has_spec       ┌────────┐
   └─────────┘                │        │──────────────────▶│  size  │
                              └────────┘                    └────────┘
                                                            (placeholder)
```

| Node type | Count | Features |
|---|---|---|
| `user` (= vehicle make+model) | 277 | learnable embedding (no x) |
| `tire` | 63 | 9-dim aggregate: `avg_rating`, `rating_std`, `n_reviews`, 6 `sub_rating` means **(recomputed from train edges only)** + `text_x ∈ ℝ³⁸⁴` from sentence-transformer over per-tire review texts |
| `brand` | 23 | learnable embedding |
| `size` | 1 | placeholder (Discount Tire data has no per-review tire size) |

Edge types: `(user, reviews, tire)` carries the 1-D rating; reverse meta-relations `(tire, rev_by, user)`, `(brand, has, tire)`, `(size, spec_of, tire)` are added for bidirectional message passing.

The `user` slot is named for backwards-compatibility with the model code; semantically it holds `(VEHICLE_MAKE, VEHICLE_MODEL)` pairs like `("FORD", "F-150")`.

---

## Architecture

### HGT Encoder

`src/models/hgt_encoder.py` — a stack of `HGTLayer`s applies type-aware multi-head attention over each meta-relation. Featureless node types (`user`, `brand`, `size`) start from learnable embeddings; the `tire` type is initialised from the 9-dim aggregate features. After encoding, every node has a `hidden_dim` vector regardless of input feature dimensions.

### Two-Tower retrieval head

`src/models/two_tower.py`

```
HGT(graph) → h_user, h_tire, h_brand, h_size
ItemTower([h_tire, h_brand_of_tire, h_size_of_tire, tire_specs, text_proj(text_x)]) → item_vec
UserTower([h_user, mean(item_vec over user's train history)])                       → user_vec
score(u, t) = (user_vec · item_vec) / temperature       (both ℓ2-normalised)
```

Both towers project into the same `out_dim` retrieval space. A learnable `temperature` scales the logits for sampled-softmax. The text embedding (per-tire review text) is fused only into the item tower via a small projection MLP.

### Loss

Default is **sampled-softmax with in-batch negatives**:

$$
\mathcal{L} = \mathrm{CE}\!\left(\frac{u_i \cdot t_j}{\tau},\ \text{target}=i\right)
$$

Optional terms (off by default):
- `--ssl-lambda` — SGL-style InfoNCE on two augmented graph views
- `--hard-neg-k` — additional hard negatives drawn from same-brand or same-size buckets
- `--loss bpr | bce` — pairwise BPR or pointwise BCE alternatives

---

## Data leakage controls

Three leakage paths were identified during the audit and explicitly closed:

| # | Leak | Fix |
|---|---|---|
| 1 | **per-edge split** placed multiple reviews of the same `(vehicle, tire)` pair into different splits — the GNN's message-passing graph then literally contained a parallel edge to the held-out target. ~48 % of held-out pairs were affected. | `ReviewEdgeSplit.from_data` now splits at the **`(user, tire)` pair level**: all reviews of the same pair land in the same split. |
| 2 | **Tire features** (`average_rating`, `rating_std`, sub-rating means) were aggregates over **all** reviews — including val/test — and entered the item tower via the spec concat. 76 % of tires had held-out ratings folded into their features. | `BPRSampler` accepts `review_df` and **recomputes the 9-dim feature matrix from train edges only**, overwriting `train_data['tire'].x` after the split. |
| 3 | Eval mask | `user_reviewed_train` is built strictly from train edges; the held-out positive itself is exempted from masking. ✓ |

Run `uv run python scripts/audit_leakage.py` to verify.

The split also reserves at least one `(vehicle, tire)` pair per tire in train, so no tire ends up outside the message-passing graph.

---

## Final results

Held-out test evaluation, best checkpoint (epoch 16) of `two_tower_vehicle_small_e50.pt`:

| Metric | @10 | @20 | @50 |
|---|---|---|---|
| Recall / HitRate | 0.34 | **0.66** | **0.94** |
| NDCG | 0.15 | 0.23 | 0.29 |

Trajectory of the leak-fix sweep (Recall@20 on test):

| Pipeline state | Test Recall@20 | Notes |
|---|---|---|
| Original user-based path | plateaued ~0.10 | extreme sparsity, loss plateau after epoch 2 |
| Vehicle node, *with* leaks | 0.76 | inflated; ~48 % of held-out pairs were duplicated in train graph |
| Vehicle node, leaks fixed, big model | 0.56 | best at epoch 1 = initialisation luck → immediate overfit (793k params, 870 train edges) |
| Vehicle node, leaks fixed, small + reg | **0.68** | best at epoch 16, smooth 0.30 → 0.68 val curve |

---

## Repository layout

```
configs/
  vehicle.yaml                              graph build config
data/processed/
  hetero_graph_vehicle.pt                   built graph + mappings + review_df
  tire_text_emb_vehicle.npy                 per-tire review-text embeddings (63 × 384)
outputs/checkpoints/
  two_tower_vehicle_small_e50.pt            best model
scripts/
  build_graph_vehicle.py                    raw JSONs → HeteroData payload
  build_review_text_embeddings.py           per-tire review concat → SentenceTransformer
  train.py                                  training entry point
  evaluate.py                               eval a checkpoint on val + test
  inference.py                              top-K recommend (existing or new vehicle)
  build_index.py                            FAISS index over item_vec
  visualize_graph.py                        graph schema + degree plots
  audit_leakage.py                          standalone leak audit
src/
  data_processing/
    graph_builder.py                        generic HeteroData assembler
    preprocessing_vehicle.py                JSON loader + train-only feature recompute
  models/
    hgt_layer.py / hgt_encoder.py           HGT building blocks
    two_tower.py                            ItemTower + UserTower + Recommender
  training/
    sampler.py                              pair-level split + BPR/softmax sampler
    trainer.py                              one full forward + loss step
    evaluation.py                           Recall@K / NDCG@K / HitRate@K
    augment.py                              SGL view augmentation
    hard_negatives.py                       brand/size bucket sampling
  losses/bpr.py
tests/
  test_hgt_forward.py
  test_split_integrity.py
  test_train_step.py
```

---

## Getting started

### 1. Environment

```bash
uv sync
source .venv/bin/activate    # macOS/Linux
```

### 2. Data

The graph is built from per-product JSON files at `/Users/.../Discount-Tire-RGCN/data/results/` (configurable in `configs/vehicle.yaml::data.raw_dir`). Each file holds the reviews for one tire product with structured `vehicle_year` / `vehicle_make` / `vehicle_model` / `overall_rating` / `sub_ratings` / `review` fields.

### 3. Build pipeline

```bash
# 1. Encode per-tire review text → 384-D vectors
uv run python scripts/build_review_text_embeddings.py

# 2. Build the heterogeneous graph (auto-attaches text_x if step 1 ran)
uv run python scripts/build_graph_vehicle.py
```

Output: `data/processed/hetero_graph_vehicle.pt` (graph + mappings + review_df + tire_df).

### 4. Train

The smaller-model + regularised configuration that produced the final result:

```bash
uv run python scripts/train.py \
  --graph-path data/processed/hetero_graph_vehicle.pt \
  --epochs 50 --steps-per-epoch 100 --batch-size 256 \
  --hidden-dim 64 --out-dim 32 --num-layers 1 --num-heads 4 \
  --dropout 0.4 --lr 5e-4 --weight-decay 5e-4 \
  --history-drop 0.3 \
  --save-path outputs/checkpoints/two_tower_vehicle_small.pt
```

The trainer:
1. Splits review edges at the `(user, tire)` pair level, keeps the train subgraph for message passing, and recomputes `tire.x` from train rows only.
2. Per step: one full HGT forward → ItemTower / UserTower → sampled-softmax over in-batch negatives.
3. Per epoch: Recall@K / NDCG@K / HitRate@K on the val split; tracks best checkpoint by `--best-metric`.
4. Restores the best weights at the end.

### 5. Evaluate / inspect / verify

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/two_tower_vehicle_small_e50.pt
uv run python scripts/audit_leakage.py
uv run python scripts/visualize_graph.py
```

### 6. Smoke tests

```bash
uv run python tests/test_hgt_forward.py
uv run python tests/test_split_integrity.py
uv run python tests/test_train_step.py
```

---

## Key design decisions

- **Vehicle replaces user, not augments it.** Per-user signals are too sparse to be useful at this dataset size; aggregating to `(make, model)` is the highest-density entity that still meaningfully constrains tire choice.
- **`sub_ratings` are stored on the per-review DataFrame**, not consumed yet. The 6-dim feedback (tread_life / wet / cornering / noise / comfort / dry) is available for future multi-task heads but the current model only uses overall rating.
- **Per-tire size is a placeholder.** The Discount-Tire dataset records a tire *model line* (e.g. "BFGoodrich KO2"), which ships in many sizes. A single placeholder `size` node preserves the schema without inventing data.
- **Capacity matters more than depth on this scale.** 870 train edges + 793k params = immediate overfit. The smaller model (159k params) learns over 16 epochs and beats the bigger one by 12 percentage points on Recall@20.
- **All architecture stayed identical** through the data-side rewrite. The model code (`hgt_encoder.py`, `two_tower.py`, `trainer.py`) was not modified during the vehicle pivot — only the data layer (`preprocessing_vehicle.py`, sampler split logic, and per-tire feature recompute).
