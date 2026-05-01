import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from model import NCF
from data import load_data


# Custom Dataset Class - tells Python how to fetch one sample
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.movies= torch.tensor(df["movieId"].values, dtype=torch.long)
        self.labels = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.labels[index]
    

def train():

    # Load Data
    train_df, test_df, n_users, n_movies, _ = load_data()
    print(f"Users: {n_users} | Movies: {n_movies}")

    # Create DataLoaders — these batch and shuffle data automatically
    train_loader = DataLoader(RatingsDataset(train_df), batch_size=512, shuffle=True)
    test_loader  = DataLoader(RatingsDataset(test_df),  batch_size=512)

    # Initialize model, loss function, optimizer
    model = NCF(n_users, n_movies)
    loss_function = nn.BCELoss()    # Binary Cross Entropy Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop - 10 epochs
    for epoch in range(15):
        model.train()
        total_loss = 0
        
        for users, movies, labels in train_loader:
            optimizer.zero_grad()   # reset gradients
            predictions = model(users, movies)
            loss = loss_function(predictions, labels)
            loss.backward()     #compute gradients
            optimizer.step()    # update weights
            total_loss += loss.item()

        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():   # no gradient needed for evaluation
            for users, movies, labels in test_loader:
                preds = model(users, movies)
                predicted = (preds >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = round(100 * correct / total, 2)
        print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc}%")

    # Save the trained model
    torch.save(model.state_dict(), "ncf_model.pt")
    print("Model saved as ncf_model.pt")

if __name__ == "__main__":
    train()