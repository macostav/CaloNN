import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

"""
Simple CNN to discern between positrons and gammas entering the PIONEER calorimeter. 
Note: a pencil source was used to generate the data.
"""

# TODO Dataset related things will probably need to be adjusted to work with the data we have

# TODO For CaloClassifier, you might need to add a transform for the images; you probably only need one to make sure the sizes work well

### DATASET DEFINITION ###
class CaloDataset(Dataset):
    def __init__(self, data_path, split = "train", transform = None):
        # Load data
        data = torch.load(data_path)

        if split == "train":
            self.X = data["X_train"]
            self.y = data["y_train"]
        elif split == "val":
            self.X = data["X_val"]
            self.y = data["y_val"]
        else:
            raise ValueError("Split must be either 'train' or 'val'")
        
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = self.X[index] # data
        y = self.y[index] # labels

        # Transform the data before returning it
        if self.transform:
            X = self.transform(X)

        return X, y

### MODELS ###

# Uses a preexisting model
class CaloClassifier(nn.Module):
    def __init__(self, num_classes = 2):
        super(CaloClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True) # The weights are already set; no need to train them
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # Get rid of final layer of model

        enet_output_size = 1280 # The output size of the model
        
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flatten the tensors into a 1D vector
            nn.Linear(enet_output_size, num_classes) # Maps the 1280 output to our two classes; line or parabola
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x) # Pattern recognition part
        output = self.classifier(x) # Last layer for classification of images
        return output
    
if __name__ == "__main__":
    # Setting up the data
    train_dataset = CaloDataset("data/prepared_dataset.pt", split="train") # 0 is gamma; 1 is positron
    val_dataset   = CaloDataset("data/prepared_dataset.pt", split="val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False) # Don't need shuffling when checking model accuracy

    # Training loop
    num_epochs = 20
    train_losses, test_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU if possible for faster training

    model = CaloClassifier()
    model.to(device)

    criterion = nn.CrossEntropyLoss() # Loss function; standard for multi-class classification; penalizes high confidence wrong predictions
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Algorithm to move to minimum of loss function; Adam is considered to be one of the best algorithms

    best_accuracy = 0.0
    accuracies = []
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): 
            for images, labels in tqdm(val_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                # Get prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = running_loss / len(val_loader.dataset)
        accuracy = correct / total * 100
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - "
        f"Train loss: {train_loss:.4f}, "
        f"Validation loss: {test_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig("pretrained_loss_evolution.png")

    plt.figure()
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.savefig("pretrained_accuracy_evolution.png")

