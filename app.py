import streamlit as st
import torch
import pandas as pd
from model import NCF
from data import load_data

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

@st.cache_resource
def load_everything():
    train_df, test_df, n_users, n_movies, movies = load_data()
    model = NCF(n_users, n_movies)
    model.load_state_dict(torch.load("ncf_model.pt", map_location="cpu"))
    model.eval()
    return model, movies, n_users, n_movies

model, movies, n_users, n_movies = load_everything()

st.title("Movie Recommendation System")
st.markdown("Powered by **Neural Collaborative Filtering** (PyTorch)")
st.divider()

# User ID input
user_id = st.slider("Select a User ID to get recommendations:", 0, n_users - 1, 0)
top_n   = st.selectbox("How many recommendations?", [5, 10, 15])

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Running neural network inference..."):
        # Score all movies for this user
        user_tensor  = torch.tensor([user_id] * n_movies, dtype=torch.long)
        movie_tensor = torch.tensor(list(range(n_movies)),  dtype=torch.long)

        with torch.no_grad():
            scores = model(user_tensor, movie_tensor).numpy()

        # Get top N movie indices
        top_indices = scores.argsort()[::-1][:top_n]

        st.subheader(f"Top {top_n} Recommendations for User {user_id}:")
        for rank, idx in enumerate(top_indices, 1):
            score = round(float(scores[idx]) * 100, 1)
            # Try to match movie name
            try:
                movie_name = movies.iloc[idx]["title"]
                genres     = movies.iloc[idx]["genres"].replace("|", " · ")
            except:
                movie_name = f"Movie {idx}"
                genres     = "N/A"

            st.markdown(f"**{rank}. {movie_name}**")
            st.caption(f"{genres} &nbsp;|&nbsp; Match score: {score}%")
            st.divider()