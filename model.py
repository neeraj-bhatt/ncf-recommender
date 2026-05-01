import torch
import torch.nn as nn



class NCF(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32, layers=[64,32,16]):
        super(NCF, self).__init__()

        # Embedding layers - each user/movie gets a vector of size embedding_dim
        # The model learns these vectors during training
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # MLP layers - takes concatenated user+movie embeddings as input
        # First layer input size = embedding_dim * 2 (user vector + movie vector)
        mlp_layers = []
        input_size = embedding_dim * 2
        
        for layer_size in layers:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2)) # to prevent overfitting
            input_size = layer_size

        self.mlp = nn.Sequential(*mlp_layers)

        # Final output layer - single score (will user like this movie?)
        self.output = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid() # squash to 0-1 probability


    def forward(self, user_ids, movie_ids):
        # Look up embeddings for user and movie IDs
        user_vec =  self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)

        # concat them into one vector
        x = torch.cat([user_vec, movie_vec], dim=1)

        # Pass through MLP
        x = self.mlp(x)

        # Get final prediction
        return self.sigmoid(self.output(x)).squeeze()