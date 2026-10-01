import numpy as np
import pandas as pd

class SVDRecommender:
    def __init__(self, n_factors=50, lr=0.005, reg=0.02, n_epochs=20):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None
        self.user_factors = None
        self.item_factors = None
        
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

    def fit(self, ratings_df):
        users = ratings_df['user_id'].unique()
        items = ratings_df['item_id'].unique()
        
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: it for i, it in enumerate(items)}
        
        n_users = len(users)
        n_items = len(items)
        
        self.global_mean = ratings_df['rating'].mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        u_indices = ratings_df['user_id'].map(self.user2idx).values
        i_indices = ratings_df['item_id'].map(self.item2idx).values
        ratings = ratings_df['rating'].values
        
        print(f"Training SVD model ({self.n_factors} factors, {self.n_epochs} epochs)...")
        for epoch in range(self.n_epochs):
            for u_idx, i_idx, r in zip(u_indices, i_indices, ratings):
                prediction = (
                    self.global_mean +
                    self.user_biases[u_idx] +
                    self.item_biases[i_idx] +
                    np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
                )
                err = r - prediction
                
                self.user_biases[u_idx] += self.lr * (err - self.reg * self.user_biases[u_idx])
                self.item_biases[i_idx] += self.lr * (err - self.reg * self.item_biases[i_idx])
                
                u_f = self.user_factors[u_idx].copy()
                self.user_factors[u_idx] += self.lr * (err * self.item_factors[i_idx] - self.reg * self.user_factors[u_idx])
                self.item_factors[i_idx] += self.lr * (err * u_f - self.reg * self.item_factors[i_idx])
                
        print("Training complete.")
        return self

    def predict(self, user_id, item_id):
        pred = self.global_mean
        u_exists = user_id in self.user2idx
        i_exists = item_id in self.item2idx
        
        if u_exists:
            pred += self.user_biases[self.user2idx[user_id]]
        if i_exists:
            pred += self.item_biases[self.item2idx[item_id]]
        if u_exists and i_exists:
            pred += np.dot(self.user_factors[self.user2idx[user_id]], self.item_factors[self.item2idx[item_id]])
            
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id, candidate_item_ids, top_n=10):
        scored_items = [(item_id, self.predict(user_id, item_id)) for item_id in candidate_item_ids]
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_n]
