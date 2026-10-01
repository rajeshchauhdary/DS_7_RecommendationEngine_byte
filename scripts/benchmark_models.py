import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import get_clean_ratings
from src.svd_recommender import SVDRecommender
from src.knn_recommender import ItemKNNRecommender
from src.metrics import compute_rmse, compute_mae

def run_benchmarks():
    ratings_df, _ = get_clean_ratings()
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    global_mean = train_df["rating"].mean()
    user_means = train_df.groupby("user_id")["rating"].mean().to_dict()
    item_means = train_df.groupby("item_id")["rating"].mean().to_dict()

    y_true = test_df["rating"].values

    # 1. Global Mean Baseline
    preds_global = np.full(len(test_df), global_mean)

    # 2. User Mean Baseline
    preds_user = [user_means.get(uid, global_mean) for uid in test_df["user_id"]]

    # 3. Item Mean Baseline
    preds_item = [item_means.get(iid, global_mean) for iid in test_df["item_id"]]

    # 4. Item-Item Cosine KNN
    print("Fitting Item-Item Cosine KNN Recommender...")
    knn = ItemKNNRecommender(k_neighbors=15)
    knn.fit(train_df)
    preds_knn = [knn.predict(row["user_id"], row["item_id"]) for _, row in test_df.iterrows()]

    # 5. Regularized SVD Recommender
    print("Fitting Regularized SVD Recommender...")
    svd = SVDRecommender(n_factors=40, lr=0.007, reg=0.04, n_epochs=25)
    svd.fit(train_df)
    preds_svd = [svd.predict(row["user_id"], row["item_id"]) for _, row in test_df.iterrows()]

    models = ["Global Mean", "User Mean", "Item Mean", "Item-Item KNN", "Regularized SVD"]
    all_preds = [preds_global, preds_user, preds_item, preds_knn, preds_svd]

    results = []
    for name, p in zip(models, all_preds):
        results.append({
            "Model": name,
            "RMSE": round(compute_rmse(y_true, p), 4),
            "MAE": round(compute_mae(y_true, p), 4)
        })

    benchmark_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("        RECOMMENDER BENCHMARK RESULTS")
    print("=" * 50)
    print(benchmark_df.to_string(index=False))
    print("=" * 50 + "\n")

    os.makedirs("outputs", exist_ok=True)
    benchmark_df.to_csv("outputs/benchmark_results.csv", index=False)
    print("Saved benchmark results to outputs/benchmark_results.csv")

if __name__ == "__main__":
    run_benchmarks()
