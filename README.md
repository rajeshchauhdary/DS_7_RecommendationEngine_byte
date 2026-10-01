# DS_7: Movie Recommendation Engine

An end-to-end Collaborative Filtering Recommendation Engine built from scratch using regularized **Singular Value Decomposition (SVD)** matrix factorization on the MovieLens 100k dataset. Includes automated ingestion, benchmark comparisons, top-N recommendation inference, cold-start handling, and unit test suites.

---

## Architecture Overview

```text
DS_7_RecommendationEngine_byte/
├── data/                       # Ingested MovieLens raw files (git-ignored)
├── models/                     # Serialized model and metadata checkpoints
├── notebooks/                  # Interactive Jupyter exploration notebooks
├── outputs/                    # Visualizations, benchmarks, and metrics
├── scripts/
│   ├── eda.py                  # Exploratory Data Analysis & visual plots
│   ├── benchmark_models.py     # Heuristic baselines vs SVD benchmark
│   ├── recommend.py            # CLI Top-N personalized inference
│   └── visualize_embeddings.py # 2D PCA latent factor space projection
├── src/
│   ├── data_loader.py          # Matrix filtering & interaction extraction
│   ├── download_data.py        # Automated archive download with SSL handling
│   ├── hybrid_recommender.py   # Bayesian popularity fallback for cold start
│   ├── metrics.py              # RMSE, MAE, Precision@K, Recall@K
│   ├── svd_recommender.py      # Core regularized SVD model with SGD
│   └── train.py                # End-to-end training and evaluation pipeline
├── tests/
│   └── test_recommender.py     # Unittest suite for validation
├── requirements.txt
└── .gitignore
```

---

## Mathematical Formulation

Predictions for rating $\hat{r}_{u,i}$ by user $u$ on item $i$ are computed as:

$$\hat{r}_{u,i} = \mu + b_u + b_i + p_u^T q_i$$

Where:
- $\mu$: Global mean rating across all observed interactions
- $b_u, b_i$: User and item deviation biases
- $p_u, q_i \in \mathbb{R}^k$: Latent factor vectors ($k = 40$)

### Optimization Objective

Parameters are learned by minimizing the regularized squared error via Stochastic Gradient Descent (SGD):

$$\min \sum_{(u, i) \in R} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda \left( b_u^2 + b_i^2 + \|p_u\|_2^2 + \|q_i\|_2^2 \right)$$

---

## Model Benchmark Evaluation

Evaluated on a 20% held-out test split of 19,858 ratings:

| Model | RMSE | MAE |
| :--- | :--- | :--- |
| Global Mean Baseline | 1.1238 | 0.9425 |
| User Mean Baseline | 1.0388 | 0.8322 |
| Item Mean Baseline | 1.0208 | 0.8159 |
| **Regularized SVD (Ours)** | **0.9185** | **0.7242** |

- **Precision@10**: 0.7083
- **Recall@10**: 0.5491

---

## Quick Start

### 1. Ingest Data & Train
```bash
python src/download_data.py
python src/train.py
```

### 2. Run Benchmarks & Visualizations
```bash
python scripts/benchmark_models.py
python scripts/eda.py
python scripts/visualize_embeddings.py
```

### 3. Generate Top-N Recommendations
```bash
python scripts/recommend.py --user_id 196 --top_n 10
```

### 4. Run Test Suite
```bash
python -m unittest -v tests/test_recommender.py
```