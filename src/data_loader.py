import os
import pandas as pd

def load_movielens_data(data_dir="data/ml-100k"):
    ratings_path = os.path.join(data_dir, "u.data")
    movies_path = os.path.join(data_dir, "u.item")

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError(f"Dataset files not found in {data_dir}. Run download script first.")

    ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
    ratings_df = pd.read_csv(ratings_path, sep="\t", names=ratings_cols, engine="python")

    movie_cols = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies_df = pd.read_csv(movies_path, sep="|", names=movie_cols, usecols=range(24), encoding="latin-1", engine="python")

    return ratings_df, movies_df

def get_clean_ratings(min_user_ratings=5, min_item_ratings=5):
    ratings_df, movies_df = load_movielens_data()

    user_counts = ratings_df["user_id"].value_counts()
    item_counts = ratings_df["item_id"].value_counts()

    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_items = item_counts[item_counts >= min_item_ratings].index

    filtered_df = ratings_df[
        ratings_df["user_id"].isin(valid_users) & 
        ratings_df["item_id"].isin(valid_items)
    ].reset_index(drop=True)

    return filtered_df, movies_df

if __name__ == "__main__":
    ratings, movies = get_clean_ratings()
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    total_ratings = len(ratings)
    sparsity = (1.0 - (total_ratings / (n_users * n_items))) * 100

    print(f"Loaded {total_ratings} ratings across {n_users} users and {n_items} items.")
    print(f"User-Item Matrix Sparsity: {sparsity:.2f}%")
