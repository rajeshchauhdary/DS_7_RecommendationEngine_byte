# DS_7: Movie Recommendation Engine

An end-to-end, production-grade Collaborative Filtering Recommendation Engine built from scratch using regularized **Singular Value Decomposition (SVD)** matrix factorization on the MovieLens 100k dataset. Includes automated data ingestion, extensive baseline benchmarking (heuristics & memory-based KNN), 3-fold cross-validation grid search, a FastAPI REST microservice, an interactive Streamlit dashboard, cold-start handling, and comprehensive unit tests.

---

## Architecture Overview

```text
DS_7_RecommendationEngine_byte/
├── app.py                      # Interactive Streamlit Web UI Dashboard
├── data/                       # Ingested MovieLens raw files (git-ignored)
├── models/                     # Serialized model and metadata checkpoints
├── notebooks/                  # Interactive Jupyter exploration notebooks
├── outputs/                    # Visualizations, tuning curves, benchmarks, reports
├── scripts/
│   ├── eda.py                  # Exploratory Data Analysis & visual plots
│   ├── benchmark_models.py     # Heuristic baselines & Item-KNN vs SVD benchmark
│   ├── tune_svd.py             # 3-Fold Cross-Validation hyperparameter grid search
│   ├── recommend.py            # CLI Top-N personalized inference
│   └── visualize_embeddings.py # 2D PCA latent factor space projection
├── src/
│   ├── api.py                  # FastAPI REST serving microservice
│   ├── data_loader.py          # Matrix filtering & interaction extraction
│   ├── download_data.py        # Automated archive download with SSL handling
│   ├── hybrid_recommender.py   # Bayesian popularity fallback for cold start
│   ├── knn_recommender.py      # Memory-based Item-Item Cosine KNN baseline
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

Parameters are learned by minimizing regularized squared error via Stochastic Gradient Descent (SGD):

$$\min \sum_{(u, i) \in R} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda \left( b_u^2 + b_i^2 + \Vert{}p_u\Vert{}_2^2 + \Vert{}q_i\Vert{}_2^2 \right)$$

---

## Model Benchmark Evaluation

Evaluated on a 20% held-out test split of 19,858 ratings:

| Model | Model Family | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| Global Mean Baseline | Heuristic | 1.1238 | 0.9425 |
| User Mean Baseline | Heuristic | 1.0388 | 0.8322 |
| Item Mean Baseline | Heuristic | 1.0208 | 0.8159 |
| Item-Item Cosine KNN | Memory-Based CF | 1.9919 | 1.6535 |
| **Regularized SVD (Ours)** | **Model-Based Latent CF** | **0.9185** | **0.7242** |

- **Precision@10**: 0.7083
- **Recall@10**: 0.5491

---

## Cross-Validation & Hyperparameter Tuning

Ran 3-fold cross-validation parameter sweeps (`scripts/tune_svd.py`):

| Factors ($k$) | Learning Rate ($\eta$) | Regularization ($\lambda$) | Mean CV RMSE |
| :--- | :--- | :--- | :--- |
| **50** | **0.010** | **0.05** | **0.9347** |
| 40 | 0.007 | 0.04 | 0.9430 |
| 20 | 0.005 | 0.02 | 0.9451 |

---

## Quick Start

### 1. Ingest Data & Train
```bash
python src/download_data.py
python src/train.py
```

### 2. Run Benchmarks, Cross-Validation & Visualizations
```bash
python scripts/benchmark_models.py
python scripts/tune_svd.py
python scripts/eda.py
python scripts/visualize_embeddings.py
```

### 3. CLI Top-N Inference
```bash
python scripts/recommend.py --user_id 196 --top_n 10
```

### 4. Launch FastAPI REST Service
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive API docs are available at `http://localhost:8000/docs`.

### 5. Launch Streamlit Web UI
```bash
streamlit run app.py
```

### 6. Run Test Suite
```bash
python -m unittest -v tests/test_recommender.py
```