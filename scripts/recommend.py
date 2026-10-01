import os
import sys
import argparse
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import get_clean_ratings

def get_recommendations(user_id, top_n=10):
    model_path = "models/svd_model.joblib"
    meta_path = "models/movies_metadata.joblib"

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Trained model or metadata not found. Run python src/train.py first.")

    model = joblib.load(model_path)
    movies_df = joblib.load(meta_path)
    ratings_df, _ = get_clean_ratings()

    # Identify items already rated by this user
    rated_items = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"])

    # Pool of candidate unrated items
    all_items = set(model.item2idx.keys())
    candidate_items = list(all_items - rated_items)

    if not candidate_items:
        print(f"User {user_id} has already rated all available items.")
        return

    scored_items = model.recommend(user_id=user_id, candidate_item_ids=candidate_items, top_n=top_n)

    movie_title_map = dict(zip(movies_df["item_id"], movies_df["title"]))

    print(f"\n================ TOP {top_n} RECOMMENDATIONS FOR USER {user_id} ================")
    print(f"{'Rank':<6}{'Item ID':<10}{'Predicted Score':<18}{'Movie Title'}")
    print("-" * 65)

    for rank, (item_id, pred_score) in enumerate(scored_items, 1):
        title = movie_title_map.get(item_id, "Unknown Title")
        print(f"{rank:<6}{item_id:<10}{pred_score:<18.2f}{title}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Top-N Movie Recommendations for a User.")
    parser.add_argument("--user_id", type=int, default=196, help="User ID to recommend items for (default: 196)")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations to display (default: 10)")
    args = parser.parse_args()

    get_recommendations(user_id=args.user_id, top_n=args.top_n)
