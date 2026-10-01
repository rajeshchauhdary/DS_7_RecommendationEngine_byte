import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import get_clean_ratings
from src.hybrid_recommender import HybridRecommender

st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="??",
    layout="wide"
)

@st.cache_resource
def load_recommender_system():
    model_path = "models/svd_model.joblib"
    meta_path = "models/movies_metadata.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error("Model artifacts not found. Please run 'python src/train.py' first.")
        st.stop()
        
    svd_model = joblib.load(model_path)
    movies_df = joblib.load(meta_path)
    ratings_df, _ = get_clean_ratings()
    
    hybrid_engine = HybridRecommender(svd_model, ratings_df, movies_df)
    movie_dict = dict(zip(movies_df["item_id"], movies_df["title"]))
    return hybrid_engine, movies_df, movie_dict, ratings_df

hybrid_engine, movies_df, movie_dict, ratings_df = load_recommender_system()

st.title("?? Movie Recommendation Engine")
st.markdown("Personalized matrix factorization (**SVD**) combined with **Bayesian Popularity Fallback** on MovieLens 100k.")

# Sidebar controls
st.sidebar.header("User & Inference Config")
user_ids = sorted(ratings_df["user_id"].unique())
selected_user = st.sidebar.selectbox("Select Existing User ID:", user_ids, index=0)
top_k = st.sidebar.slider("Number of Recommendations (Top-K):", min_value=5, max_value=25, value=10)

allow_custom = st.sidebar.checkbox("Simulate New / Cold-Start User")
if allow_custom:
    custom_uid = st.sidebar.number_input("Custom User ID (Unseen):", min_value=9999, max_value=99999, value=9999)
    target_user = int(custom_uid)
else:
    target_user = int(selected_user)

# Tabbed Layout
tab1, tab2, tab3 = st.tabs(["?? Recommendations", "?? Catalog & User History", "?? Latent Space Explorer"])

with tab1:
    st.subheader(f"Top-{top_k} Movie Recommendations for User {target_user}")
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Calculating personalized scores..."):
            recs = hybrid_engine.recommend(user_id=target_user, top_n=top_k)
            
            rec_data = []
            for rank, (item_id, score, strategy) in enumerate(recs, 1):
                rec_data.append({
                    "Rank": rank,
                    "Movie Title": movie_dict.get(item_id, "Unknown Title"),
                    "Predicted Rating": f"{score:.2f} / 5.00",
                    "Recommendation Strategy": strategy
                })
            
            rec_df = pd.DataFrame(rec_data)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Historical Interactions for Selected User")
    user_history = ratings_df[ratings_df["user_id"] == target_user].sort_values(by="rating", ascending=False)
    
    if not user_history.empty:
        user_history["Movie Title"] = user_history["item_id"].map(movie_dict)
        st.write(f"Total Movies Rated by User {target_user}: **{len(user_history)}**")
        st.dataframe(
            user_history[["item_id", "Movie Title", "rating"]].rename(columns={"rating": "Actual Rating"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No historical ratings found. System will serve Bayesian Popularity fallbacks.")

with tab3:
    st.subheader("Item Latent Factor Projections (PCA 2D)")
    scatter_path = "outputs/item_latent_space.png"
    if os.path.exists(scatter_path):
        st.image(scatter_path, caption="2D PCA Projection of Top SVD Item Embeddings", use_container_width=True)
    else:
        st.info("Run 'python scripts/visualize_embeddings.py' to generate PCA visualizations.")
