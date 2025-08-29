import numpy as np
import torch
from sklearn.model_selection import train_test_split

"""
Takes all of the data we have for our CNN, separates it into different datasets, and labels it. Gamma: 0, Positron: 1.
"""

if __name__ == "__main__":
    # Reading numpy files and preparing labels
    gamma_data = np.load("data/gamma_data.npy")
    n_gamma_events = len(gamma_data[:,0])
    gamma_labels = np.zeros(n_gamma_events, dtype = int)

    positron_data = np.load("data/positron_data.npy")
    n_positron_events = len(gamma_data[:,0])
    positron_labels = np.ones(n_positron_events, dtype = int)
    
    # TODO Concatenate into one large data pool

    # TODO Separate into training and validation
    
    # TODO Transform into tensors so that torch can accept them

    # TODO Save datasets separately