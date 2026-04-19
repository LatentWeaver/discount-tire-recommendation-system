# GNN-based Tire Recommendation System  

An advanced, GNN-based tire recommendation engine that transcends traditional collaborative filtering by deploying **Heterogeneous Graph Neural Networks (HGT)**, **Deep Embedding Clustering**, and **Contrastive Learning**. It is specifically designed to decode complex user feedback, isolate hidden complaint patterns, and deliver highly contextual, complementary product recommendations.

---

## System Architecture

### Heterogeneous Graph Schema
To fully leverage **HGT**, tires are not treated as isolated points; instead, we construct a rich, interconnected network mapping out interactions and specifications.

- **Node Types**:
    - `User`: Contains user attributes (e.g., `user_id`).
    - `Tire/Item`: Contains continuous/categorical features (e.g., `price`, `average_rating`, `UTQG`).
    - `Brand`: Captures brand loyalties and preferences (e.g., "Michelin", "Landspider").
    - `Size`: The most critical node for tire recommendation (e.g., "235/40R18"). Tires with the same specifications are naturally clustered through this node.
- **Edge Types (Meta-Relations)**:
    - `User -[reviews]-> Tire`: Weighted by the review rating.
    - `Tire -[belongs_to]-> Brand`: Brand affiliation.
    - `Tire -[has_spec]-> Size`: Specification grouping.

### Feature Extraction Layer (HGT Encoder)
The **HGT Network** addresses the graph's heterogeneity by dynamically learning the importance of different meta-relations (e.g., recognizing that "Size" often carries a higher weight than "Brand").
- **Output:** High-dimensional Node Embeddings ($h_{user}$ and $h_{tire}$).
- **Advantage:** HGT automatically uncovers deep latent associations, such as learning that "a specific sized tire is vastly more popular within a certain brand."

### Intermediate Layer: Dual-Path Processing
This is the core architectural optimization of the system. The pipeline splits into two synchronized branches:

- **Path A: Deep Clustering (Intent Discovery).** Discovers latent tire profiles (e.g., "Economy/High-Mileage," "Performance/Sport," "Off-Road/All-Terrain") via an alternating DeepCluster procedure:
    1. **Feature preprocessing.** Take $h_{tire}$ from the HGT encoder and apply **PCA → whitening → $\ell_2$-normalization** before clustering.
    2. **k-means (every $N$ epochs).** Cluster the preprocessed embeddings with k-means to produce pseudo-labels $\hat{y}_{tire}$. $k$ is set to roughly $10\times$ the number of intended tire profiles (e.g., $k \approx 50$ for ~5 target categories).
    3. **Classifier head (clustering MLP).** A small MLP head $g_W(h_{tire})$ is trained with cross-entropy against $\hat{y}_{tire}$. Gradients flow back through the HGT encoder, so the encoder learns embeddings that are naturally cluster-friendly.
    4. **Trivial-solution safeguards.**
        - **Empty-cluster reassignment.** If a cluster collapses during k-means, a non-empty centroid is copied with small Gaussian noise and the points are re-split.
        - **Uniform sampling over pseudo-labels** (or equivalently, inverse-frequency loss weighting) to prevent all tires from collapsing into one dominant cluster.
    - **Output (training):** soft cluster predictions $g_W(h_{tire})$ used for the auxiliary CE loss.
    - **Output (inference):** the cluster probability distribution $C_{tire} = \mathrm{softmax}(g_W(h_{tire}))$, exposed to the Fusion MLP as an additional signal. The classifier head is **kept** at inference (rather than discarded) so the recommendation MLP gets an interpretable "intent" channel.

- **Path B: Feature Transformation.** Applies non-linear transformations to $h_{user}$ and $h_{tire}$ to prepare them for dense vector matching.

### Output Layer: Fusion & Ranking MLP
This final layer is the confluence point, aggregating all information to produce a **ranking score** used for top-K recommendation.

$$ s(u, t) = \text{MLP}(\underbrace{h_{user} \oplus h_{tire}}_{\text{Individual Features}} \oplus \underbrace{C_{tire}}_{\text{Cluster Group Features}}) $$

The output is a raw scalar — higher means "more relevant for this user".
#### Training Objective: BPR (Pairwise Ranking)
Per training step, sample triplets $(u, t^+, t^-)$ where:
- $t^+$ — a tire the user actually engaged with (review rating $\geq 4$).
- $t^-$ — a randomly sampled tire the user has not reviewed (preferably matching the same `size` to avoid trivial negatives).

Loss:

$$ \mathcal{L}_{BPR} = -\frac{1}{|B|} \sum_{(u, t^+, t^-) \in B} \log \sigma\big(s(u, t^+) - s(u, t^-)\big) $$

#### Contrastive Loss — "move users away from bad products"
For users who have BOTH a good (rating $\geq$ threshold) AND a disliked (rating $<$ threshold) review, sample triplets $(u, t_{\text{good}}, t_{\text{disliked}})$ where both tires are items the user personally owned. The loss explicitly teaches the model to rank good purchases above the user's own disappointments:

$$ \mathcal{L}_{contrast} = -\frac{1}{|B|} \sum \log \sigma\big(s(u, t_{\text{good}}) - s(u, t_{\text{disliked}})\big) $$

Unlike random BPR negatives, $t_{\text{disliked}}$ is a tire the user has actually rejected — the strongest available signal for the "transfer away from bad products" goal.

#### Combined Objective

$$ \mathcal{L}_{total} = \mathcal{L}_{BPR} + \lambda_{c} \cdot \mathcal{L}_{cluster} + \lambda_{\text{con}} \cdot \mathcal{L}_{contrast} $$

Defaults: $\lambda_{c} = 0.5$, $\lambda_{\text{con}} = 0.3$. Set $\lambda_{\text{con}} = 0$ to ablate the contrastive term.

#### Evaluation Metrics

- **Recall@K** — fraction of held-out positive tires recovered in the top-K predictions.
- **NDCG@K** — rewards placing positives near the top of the ranked list.
- **HitRate@K** — at least one relevant tire in top-K.

Standard $K \in \{10, 20, 50\}$.

#### Outputs
1. **Ranking Score $s(u, t)$** — used to rank candidate tires per user; the top-K become the recommendation.
2. **Cluster Tagging $C_{tire}$** — reused for backend analytics and user profiling (e.g., building a persona that strongly favors the "Budget" cluster).

### Model Architecture

![Model Architecture](src/img/Model%20Architecture.png)
---

## Data Schema
![Data Schema](src/img/Data%20Schema.png)

---

## Getting Started

### 1. Data Setup

Place the raw JSONL dataset in the `data/raw/` folder and create the `data/processed/` folder for output:

```bash
mkdir -p data/raw data/processed
```

Then move or copy your dataset file into `data/raw/`:

```
data/
├── raw/
│   └── combined_tire_data_15k_cleaned.jsonl   # <-- put your dataset here
└── processed/                                  # graph output will be saved here
```

### 2. Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for incredibly fast dependency management. All required packages are declared in `pyproject.toml`.

To create a virtual environment and automatically install all packages specified in `pyproject.toml`, run:

```bash
# Create the virtual environment (.venv) and install dependencies
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Build the Heterogeneous Graph

```bash
uv run python scripts/build_graph.py
```

This reads the JSONL data, constructs the PyG `HeteroData` graph, runs sanity checks, and saves the result to `data/processed/hetero_graph.pt`.

### 4. Visualize the Graph

```bash
uv run python scripts/visualize_graph.py
```

Generates four figures in `outputs/figures/`:
- `graph_schema.png` -- node and edge type overview
- `subgraph_sample.png` -- sampled neighbourhood around a few users
- `degree_distributions.png` -- edge degree histograms per node type
- `tire_feature_distributions.png` -- distribution of each tire feature

### 5. Train the Recommender

```bash
uv run python scripts/train.py
```

Runs the full pipeline `HGT encoder → IntermediateLayer (Path A + B) → FusionMLP` with the joint objective $\mathcal{L}_{BPR} + \lambda \cdot \mathcal{L}_{cluster}$.

Each epoch:
1. **Pseudo-label refresh** (every `--refresh-every` epochs): one full graph forward → snapshot `h_tire` → PCA → whiten → ℓ2-norm → k-means → empty-cluster repair → frozen labels for the next $N$ epochs.
2. **Training steps**: per step the BPR sampler draws $(u, t^+, t^-)$ triplets, the model scores both, and the optimizer minimizes $\mathcal{L}_{BPR} + \lambda \cdot \mathcal{L}_{cluster}$.
3. **Eval** (every `--eval-every` epochs): Recall@K / NDCG@K / HitRate@K on the held-out val split, with the user's training positives masked out.

Common knobs:

```bash
uv run python scripts/train.py \
  --epochs 30 --batch-size 2048 --lr 5e-4 \
  --num-clusters 50 --cluster-lambda 0.5 \
  --refresh-every 2 --rating-threshold 4.0
```

The script prints per-epoch losses (`L_bpr`, `L_cluster`) and final test-set top-K metrics.

### Smoke Tests

```bash
uv run python tests/test_hgt_forward.py        # HGT encoder forward
uv run python tests/test_intermediate_layer.py # Intermediate layer + DeepCluster refresh
uv run python tests/test_train_step.py         # End-to-end: 1 refresh + 3 train steps + mini eval
```

### 6. Recommend (Inductive GNN Inference)

```bash
# Existing user (by index):
uv run python scripts/inference.py --user 42 --k 10

# New user with structured preferences (Inductive GNN):
uv run python scripts/inference.py --user new \
  --brand "Michelin,Continental" \
  --size "235/40R18" \
  --budget-min 100 --budget-max 250 \
  --min-treadwear 400 \
  --traction A --temperature A \
  --k 10
```

For **new users**, the script uses Inductive GNN inference:
1. Finds tires matching the preference filters (AND logic).
2. Injects a temporary user node into the graph with preference edges to matching tires.
3. Runs the full HGT → Intermediate → FusionMLP pipeline on the augmented graph.
4. Returns top-K scored tires with names and cluster IDs.

| Filter | Flag | Example |
|--------|------|---------|
| Brand | `--brand` | `"Michelin,Continental"` |
| Size | `--size` | `"235/40R18"` |
| Budget | `--budget-min/max` | `100`, `250` |
| Treadwear | `--min-treadwear` | `400` |
| Traction | `--traction` | `A` (grades: AA > A > B > C) |
| Temperature | `--temperature` | `A` (grades: A > B > C) |

