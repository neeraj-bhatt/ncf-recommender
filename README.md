# 🎬 Neural Collaborative Filtering — Movie Recommender System

A deep learning-based movie recommendation system built with **PyTorch**, trained on the **MovieLens dataset**. Uses Neural Collaborative Filtering (NCF) with learned user and item embeddings passed through a multi-layer perceptron to predict user preferences.

🔗 **[Live Demo](https://ncf-recommender-neeraj.streamlit.app/)** &nbsp;|&nbsp; ⭐ Star this repo if you found it useful!

---

## 🧠 How It Works

Traditional recommendation systems use simple similarity metrics (cosine similarity, matrix factorization). NCF replaces the dot product with a neural network, allowing the model to learn **non-linear user-item interactions**.

```
User ID ──► User Embedding ──┐
                              ├──► Concatenate ──► MLP (64→32→16) ──► Sigmoid ──► Score
Movie ID ──► Item Embedding ──┘
```

1. Each user and movie is mapped to a dense embedding vector (learned during training)
2. User and movie vectors are concatenated
3. Passed through a 3-layer MLP with ReLU activations and Dropout
4. Sigmoid output gives a probability score (will this user like this movie?)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Dataset | MovieLens Latest Small |
| Users | ~610 |
| Movies | ~9,700 |
| Ratings | ~100,000 |
| Test Accuracy | 72%+ |
| Loss Function | Binary Cross Entropy |
| Optimizer | Adam (lr=0.001) |
| Training | Early stopping (patience=3) |

---

## 🛠 Tech Stack

- **Model:** PyTorch — `nn.Embedding`, `nn.Linear`, `nn.ReLU`, `nn.Dropout`
- **Data:** Pandas, NumPy, Scikit-learn
- **Frontend:** Streamlit
- **Dataset:** [MovieLens Latest Small](https://grouplens.org/datasets/movielens/)
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

```
ncf-recommender/
├── data.py          # Dataset download, preprocessing, train/test split
├── model.py         # NCF neural network architecture
├── train.py         # Training loop, evaluation, early stopping, model saving
├── app.py           # Streamlit web app for inference
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/neeraj-bhatt/ncf-recommender.git
cd ncf-recommender
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model**
```bash
python train.py
# Downloads MovieLens dataset automatically
# Trains for up to 50 epochs with early stopping
# Saves best model as ncf_model.pt
```

**5. Launch the app**
```bash
streamlit run app.py
```

---

## 💡 Features

- 🔍 Real-time movie recommendations for any user ID
- 🎯 Adjustable number of recommendations (Top 5 / 10 / 15)
- 🏷️ Displays movie title, genre, and match score
- 📦 Batch inference across all 9,700+ movies in one forward pass
- 🛑 Early stopping to prevent overfitting — saves best checkpoint automatically

---

## ⚠️ Limitations & Future Work

- Model uses **implicit feedback** (rating ≥ 3.5 = liked) — does not model rating magnitude
- Cold start problem — cannot recommend for new users/movies not seen during training
- Future improvement: incorporate movie metadata (genre, year) as additional features
- Future improvement: use BPR (Bayesian Personalized Ranking) loss for better ranking
