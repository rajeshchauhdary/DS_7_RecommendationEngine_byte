import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_clean_ratings
from src.svd_recommender import SVDRecommender
from src.metrics import compute_rmse, compute_mae, precision_recall_at_k

def run_training():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Loading filtered interaction data...")
    ratings_df, movies_df = get_clean_ratings()

    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    model = SVDRecommender(n_factors=40, lr=0.007, reg=0.04, n_epochs=25)
    model.fit(train_df)

    print("Evaluating predictions on test partition...")
    predictions = []
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        rating = float(row["rating"])
        pred_r = model.predict(uid, iid)
        
        predictions.append((uid, iid, rating, pred_r))
        y_true.append(rating)
        y_pred.append(pred_r)

    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    p_10, r_10 = precision_recall_at_k(predictions, k=10, threshold=3.5)

    metrics_summary = f"""================ SVD EVALUATION REPORT ================
Test RMSE        : {rmse:.4f}
Test MAE         : {mae:.4f}
Precision@10     : {p_10:.4f}
Recall@10        : {r_10:.4f}
======================================================
"""
    print(metrics_summary)

    with open("outputs/evaluation_report.txt", "w") as f:
        f.write(metrics_summary)

    joblib.dump(model, "models/svd_model.joblib")
    joblib.dump(movies_df, "models/movies_metadata.joblib")
    print("Model and movie metadata successfully saved in models/")

if __name__ == "__main__":
    run_training()
