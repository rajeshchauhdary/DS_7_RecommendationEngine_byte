import numpy as np
import pandas as pd

class HybridRecommender:
    def __init__(self, svd_model, ratings_df, movies_df, min_votes=20):
        self.svd = svd_model
        self.ratings = ratings_df
        self.movies = movies_df
        self.min_votes = min_votes
        
        # Bayesian average calculation for fallback
        c = ratings_df['rating'].mean()
        stats = ratings_df.groupby('item_id').agg({'rating': ['count', 'mean']})
        stats.columns = ['count', 'mean']
        
        stats['weighted_score'] = (
            (stats['count'] / (stats['count'] + self.min_votes)) * stats['mean'] +
            (self.min_votes / (stats['count'] + self.min_votes)) * c
        )
        self.popular_items = stats.sort_values(by='weighted_score', ascending=False)

    def recommend(self, user_id, top_n=10):
        # Known user with latent factors: personalized SVD recommendation
        if user_id in self.svd.user2idx:
            rated_items = set(self.ratings[self.ratings['user_id'] == user_id]['item_id'])
            candidates = [iid for iid in self.svd.item2idx.keys() if iid not in rated_items]
            return [(iid, self.svd.predict(user_id, iid), "SVD_Collaborative") 
                    for iid, _ in self.svd.recommend(user_id, candidates, top_n=top_n)]
        
        # Cold-start user: weighted popularity fallback
        popular_top = self.popular_items.head(top_n)
        return [(int(idx), float(row['weighted_score']), "ColdStart_BayesianFallback") 
                for idx, row in popular_top.iterrows()]
