import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import requests, zipfile, io, os


# Downloads Movielens dataset
def download_movielens():
    if not os.path.exists("data/ratings.csv"):
        os.makedirs("data", exist_ok=True)
        print("Downloading MovieLens 1M dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("data/")
        print("Done!")


# Load the Dataset
def load_data():
    download_movielens()

    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")

    # Encode user and movie IDs to sequential integers
    # model needs 0, 1, 2... not arbitrary IDs
    ratings["userId"] = ratings["userId"].astype("category").cat.codes
    ratings["movieId"] = ratings["movieId"].astype("category").cat.codes

    # Binary feedback - did user rate this? (0=no, 1=yes)
    ratings["rating"] = (ratings["rating"] >= 3.5).astype(float)

    n_users = ratings["userId"].nunique()
    n_movies = movies["movieId"].nunique()

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    return train, test, n_users, n_movies, movies



