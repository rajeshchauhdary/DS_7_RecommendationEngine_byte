import os
import sys
import joblib
from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import get_clean_ratings
from src.hybrid_recommender import HybridRecommender

app = FastAPI(
    title="Movie Recommendation Engine API",
    description="REST API for serving personalized collaborative filtering recommendations via SVD with Bayesian cold-start fallback.",
    version="1.0.0"
)

hybrid_engine = None
movie_dict = {}

@app.on_event("startup")
def load_artifacts():
    global hybrid_engine, movie_dict
    model_path = "models/svd_model.joblib"
    meta_path = "models/movies_metadata.joblib"

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise RuntimeError("Model or metadata artifacts not found. Run python src/train.py first.")

    svd_model = joblib.load(model_path)
    movies_df = joblib.load(meta_path)
    ratings_df, _ = get_clean_ratings()

    movie_dict = dict(zip(movies_df["item_id"], movies_df["title"]))
    hybrid_engine = HybridRecommender(svd_model, ratings_df, movies_df)
    print("Inference engine ready.")

class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    title: str
    predicted_score: float
    strategy: str

class RecommendationResponse(BaseModel):
    user_id: int
    total_recommendations: int
    recommendations: List[RecommendationItem]

@app.get("/health", tags=["Status"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": hybrid_engine is not None,
        "total_catalog_items": len(movie_dict)
    }

@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Inference"])
def get_recommendations(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=50, description="Number of items to recommend")
):
    if hybrid_engine is None:
        raise HTTPException(status_code=503, detail="Model engine not loaded")

    raw_recs = hybrid_engine.recommend(user_id=user_id, top_n=top_n)

    formatted_recs = []
    for rank, (item_id, score, strategy) in enumerate(raw_recs, 1):
        formatted_recs.append(
            RecommendationItem(
                rank=rank,
                item_id=item_id,
                title=movie_dict.get(item_id, "Unknown Title"),
                predicted_score=round(score, 3),
                strategy=strategy
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        total_recommendations=len(formatted_recs),
        recommendations=formatted_recs
    )
