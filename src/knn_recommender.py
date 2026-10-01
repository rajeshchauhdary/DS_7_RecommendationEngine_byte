import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemKNNRecommender:
    def __init__(self, k_neighbors=10):
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.user_item_matrix = None
        self.user_means = None
        self.global_mean = 0.0
        self.item_ids = []
        self.item2idx = {}
        self.user2idx = {}

    def fit(self, ratings_df):
        self.global_mean = ratings_df['rating'].mean()
        pivot = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating')
        self.user_means = pivot.mean(axis=1)
        
        # Center ratings around each user's mean to account for rating bias
        pivot_centered = pivot.sub(self.user_means, axis=0).fillna(0)
        
        self.user_item_matrix = pivot.fillna(0).values
        self.item_ids = list(pivot.columns)
        self.item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.user2idx = {uid: idx for idx, uid in enumerate(pivot.index)}
        
        # Compute item-item cosine similarity on centered interactions
        item_vectors = pivot_centered.values.T
        self.item_similarity = cosine_similarity(item_vectors)
        np.fill_diagonal(self.item_similarity, 0.0)
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user2idx or item_id not in self.item2idx:
            return float(self.global_mean)

        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[item_id]

        user_ratings = self.user_item_matrix[u_idx]
        rated_mask = user_ratings > 0

        sims = self.item_similarity[i_idx, rated_mask]
        ratings = user_ratings[rated_mask]

        if len(sims) == 0 or np.sum(np.abs(sims)) == 0:
            return float(self.user_means.iloc[u_idx]) if u_idx < len(self.user_means) else float(self.global_mean)

        top_k_idx = np.argsort(np.abs(sims))[-self.k_neighbors:]
        selected_sims = sims[top_k_idx]
        selected_ratings = ratings[top_k_idx]

        denom = np.sum(np.abs(selected_sims))
        if denom == 0:
            return float(self.global_mean)

        pred = np.sum(selected_sims * selected_ratings) / denom
        return float(np.clip(pred, 1.0, 5.0))
