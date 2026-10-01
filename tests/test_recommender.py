import unittest
import numpy as np
import pandas as pd
from src.metrics import compute_rmse, compute_mae, precision_recall_at_k
from src.svd_recommender import SVDRecommender
from src.hybrid_recommender import HybridRecommender

class TestRecommenderEngine(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = [4.0, 3.0, 5.0]
        y_pred = [4.0, 3.0, 5.0]
        self.assertEqual(compute_rmse(y_true, y_pred), 0.0)
        self.assertEqual(compute_mae(y_true, y_pred), 0.0)

    def test_svd_prediction_bounds(self):
        dummy_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [10, 20, 10, 30, 20],
            'rating': [5.0, 4.0, 3.0, 2.0, 4.0]
        })
        model = SVDRecommender(n_factors=5, n_epochs=5)
        model.fit(dummy_data)
        
        pred = model.predict(1, 30)
        self.assertTrue(1.0 <= pred <= 5.0)

    def test_hybrid_cold_start(self):
        dummy_ratings = pd.DataFrame({
            'user_id': [1, 1, 2],
            'item_id': [101, 102, 101],
            'rating': [5.0, 4.0, 5.0]
        })
        dummy_movies = pd.DataFrame({
            'item_id': [101, 102],
            'title': ['Film A', 'Film B']
        })
        model = SVDRecommender(n_factors=2, n_epochs=2)
        model.fit(dummy_ratings)
        
        hybrid = HybridRecommender(model, dummy_ratings, dummy_movies, min_votes=1)
        recs = hybrid.recommend(user_id=99999, top_n=2)
        
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0][2], "ColdStart_BayesianFallback")

if __name__ == '__main__':
    unittest.main()
