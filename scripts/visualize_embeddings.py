import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def plot_item_embeddings(top_k=25, output_path="outputs/item_latent_space.png"):
    model_path = "models/svd_model.joblib"
    meta_path = "models/movies_metadata.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model or metadata missing. Run python src/train.py first.")
        
    model = joblib.load(model_path)
    movies_df = joblib.load(meta_path)
    movie_dict = dict(zip(movies_df["item_id"], movies_df["title"]))
    
    # Extract top popular items for clear visualization
    item_factors = model.item_factors[:top_k]
    item_ids = [model.idx2item[i] for i in range(top_k)]
    item_titles = [movie_dict.get(iid, f"Item {iid}")[:20] for iid in item_ids]
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(item_factors)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], color="#1f77b4", alpha=0.8, edgecolors="k", s=80)
    
    for i, title in enumerate(item_titles):
        plt.annotate(
            title,
            (coords[i, 0], coords[i, 1]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )
        
    plt.title("SVD Movie Latent Feature Space (PCA 2D Projection)")
    plt.xlabel("Latent Component 1")
    plt.ylabel("Latent Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Embedding visualization saved to {output_path}")

if __name__ == "__main__":
    plot_item_embeddings()
