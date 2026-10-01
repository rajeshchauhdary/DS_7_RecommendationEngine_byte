import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import get_clean_ratings
from src.svd_recommender import SVDRecommender
from src.metrics import compute_rmse

def evaluate_fold(train_df, val_df, n_factors, lr, reg, n_epochs=15):
    model = SVDRecommender(n_factors=n_factors, lr=lr, reg=reg, n_epochs=n_epochs)
    model.fit(train_df)
    
    val_preds = [model.predict(r['user_id'], r['item_id']) for _, r in val_df.iterrows()]
    return compute_rmse(val_df['rating'].values, val_preds)

def tune_hyperparameters():
    ratings_df, _ = get_clean_ratings()
    
    # Stratified/shuffled 3-fold split for quick parameter sweeps
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    param_grid = [
        {"n_factors": 20, "lr": 0.005, "reg": 0.02},
        {"n_factors": 40, "lr": 0.007, "reg": 0.04},
        {"n_factors": 50, "lr": 0.010, "reg": 0.05}
    ]
    
    print("\nStarting Cross-Validation Grid Search over SVD Configurations...")
    results = []
    
    for params in param_grid:
        fold_rmses = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(ratings_df)):
            train_fold = ratings_df.iloc[train_idx]
            val_fold = ratings_df.iloc[val_idx]
            rmse = evaluate_fold(train_fold, val_fold, params["n_factors"], params["lr"], params["reg"], n_epochs=12)
            fold_rmses.append(rmse)
        
        mean_rmse = float(np.mean(fold_rmses))
        results.append({
            "Factors": params["n_factors"],
            "LearningRate": params["lr"],
            "Regularization": params["reg"],
            "Mean_CV_RMSE": round(mean_rmse, 4)
        })
        print(f"Params: {params} --> Mean CV RMSE: {mean_rmse:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="Mean_CV_RMSE")
    print("\n" + "=" * 55)
    print("           HYPERPARAMETER TUNING SUMMARY")
    print("=" * 55)
    print(results_df.to_string(index=False))
    print("=" * 55 + "\n")
    
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/svd_tuning_results.csv", index=False)
    
    # Generate tuning comparison plot
    plt.figure(figsize=(8, 5))
    labels = [f"F:{r['Factors']} | LR:{r['LearningRate']} | Reg:{r['Regularization']}" for _, r in results_df.iterrows()]
    plt.barh(labels, results_df["Mean_CV_RMSE"], color="#2ca02c")
    plt.xlabel("Mean 3-Fold CV RMSE")
    plt.title("SVD Hyperparameter Configuration Comparison")
    plt.xlim(min(results_df["Mean_CV_RMSE"]) - 0.05, max(results_df["Mean_CV_RMSE"]) + 0.05)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/hyperparameter_tuning.png", dpi=300)
    plt.close()
    print("Saved outputs/svd_tuning_results.csv and outputs/hyperparameter_tuning.png")

if __name__ == "__main__":
    tune_hyperparameters()
