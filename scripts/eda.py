import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import get_clean_ratings

def run_eda(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    ratings_df, movies_df = get_clean_ratings()

    # 1. Rating Frequency Distribution
    plt.figure(figsize=(7, 4))
    sns.countplot(x="rating", hue="rating", data=ratings_df, palette="Blues_d", legend=False)
    plt.title("MovieLens 100k - Rating Frequency Distribution")
    plt.xlabel("Rating (1-5 Stars)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=300)
    plt.close()

    # 2. User Activity Distribution
    user_counts = ratings_df.groupby("user_id")["rating"].count()
    plt.figure(figsize=(7, 4))
    sns.histplot(user_counts, bins=40, kde=True, color="#2b5c8f")
    plt.title("Distribution of Ratings Given per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "user_interaction_dist.png"), dpi=300)
    plt.close()

    print(f"EDA plots saved to {output_dir}/")

if __name__ == "__main__":
    run_eda()
