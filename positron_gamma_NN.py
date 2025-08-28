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

# Requires an image shape rather than a 1D array
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
    
class SimpleCaloClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCaloClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)  
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * (1894 // 4), 128)  # after two pools
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (N, 32, 947)
        x = self.pool(self.relu(self.conv2(x)))   # (N, 64, 473)
        x = x.view(x.size(0), -1)                 # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CaloMLPClassifier(nn.Module):
    def __init__(self, input_dim=1894, num_classes=2):
        super(CaloMLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),   # first dense layer
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),         # second dense layer
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),          # bottleneck layer
            nn.ReLU(),

            nn.Linear(64, num_classes)   # output logits
        )

    def forward(self, x):
        # Ensure input is (batch_size, input_dim)
        if x.ndim > 2:  
            x = x.view(x.size(0), -1)
        return self.layers(x)
    
class CaloCNNClassifier(nn.Module):
    def __init__(self, input_length=1894, num_classes=2):
        super(CaloCNNClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),  # keep length
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),   # halve length → ~947

            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),   # halve length → ~473

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),   # halve length → ~236
        )

        # compute flattened size after convolutions
        conv_out_size = 128 * (input_length // (2**3))  # three pools → /8

        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Expecting shape: (batch, 1, input_length)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)
    
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

    model = CaloCNNClassifier()
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

