import numpy as np
import torch
from sklearn.model_selection import train_test_split

"""
Takes all of the data we have for our NN, separates it into different datasets, and labels it. Gamma: 0, Positron: 1.
"""

if __name__ == "__main__":
    # Reading numpy files and preparing labels
    gamma_data = np.load("gamma_cnn_data.npy")
    n_gamma_events = len(gamma_data[:,0])
    gamma_labels = np.zeros(n_gamma_events, dtype = int)

    positron_data = np.load("positron_cnn_data.npy")
    n_positron_events = len(gamma_data[:,0])
    positron_labels = np.ones(n_positron_events, dtype = int)
    
    # Concatenate
    X = np.vstack([gamma_data, positron_data])   # shape (100000, 1894)
    y = np.concatenate([gamma_labels, positron_labels])  # shape (100000,)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Separate into different sets
    
    # Transform into tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor   = torch.tensor(y_val, dtype=torch.long)

    # Save dataset
    torch.save({
    "X_train": X_train_tensor,
    "y_train": y_train_tensor,
    "X_val": X_val_tensor,
    "y_val": y_val_tensor
    }, "prepared_dataset.pt")
