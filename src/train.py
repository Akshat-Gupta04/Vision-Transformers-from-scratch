# train.py
import timeit
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd

# Import hyperparameters from hyperparams.py
from src.hyperparam import (
    random_seed, batch_size, epochs, learning_rate, num_classes,
    patch_size, img_size, in_channels, num_heads, dropout, hidden_dim,
    adam_weight_decay, adam_betas, activation, num_encoders, embed_dim, device
)

# Import the VisionTransformer model from model.py
from src.model import VisionTransformer

# Import dataset classes from dataset.py
from src.dataset import MNISTTrainDataset, MNISTValDataset, MNISTSubmitDataset

# =============================================================================
# Load DataFrames
# =============================================================================
# Adjust the file paths as needed; these CSVs should contain your MNIST data.
# For train and validation, the first column is assumed to be labels.
train_df = pd.read_csv("train.csv")   # e.g., shape: (N, 785) with first column as label
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")     # test dataframe with only pixel columns

# =============================================================================
# Create Dataset Instances
# =============================================================================
train_dataset = MNISTTrainDataset(
    images=train_df.iloc[:, 1:].values.astype(np.uint8),
    labels=train_df.iloc[:, 0].values,
    indicies=train_df.index.values
)

val_dataset = MNISTValDataset(
    images=val_df.iloc[:, 1:].values.astype(np.uint8),
    labels=val_df.iloc[:, 0].values,
    indicies=val_df.index.values
)

test_dataset = MNISTSubmitDataset(
    images=test_df.values.astype(np.uint8),
    indicies=test_df.index.values
)

# =============================================================================
# Create DataLoaders
# =============================================================================
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# =============================================================================
# Instantiate Model, Criterion & Optimizer
# =============================================================================
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes,
    embed_dim=embed_dim,
    num_encoders=num_encoders,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    dropout=dropout
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=adam_betas, lr=learning_rate, weight_decay=adam_weight_decay)

# =============================================================================
# Training & Validation Loop
# =============================================================================
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start = timeit.default_timer()
for epoch in tqdm(range(epochs), desc="Epochs", position=0, leave=True):
    # --- Training ---
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0
    for idx, batch in enumerate(tqdm(train_dataloader, desc="Training", position=1, leave=False)):
        imgs = batch["image"].float().to(device)
        labels = batch["label"].long().to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Collect predictions and loss
        train_running_loss += loss.item()
        train_labels.extend(labels.cpu().detach().numpy())
        train_preds.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy())
        
    train_loss = train_running_loss / (idx + 1)
    train_acc = sum(1 for p, t in zip(train_preds, train_labels) if p == t) / len(train_labels)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # --- Validation ---
    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader, desc="Validation", position=1, leave=False)):
            imgs = batch["image"].float().to(device)
            labels = batch["label"].long().to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            val_labels.extend(labels.cpu().detach().numpy())
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy())
            
    val_loss = val_running_loss / (idx + 1)
    val_acc = sum(1 for p, t in zip(val_preds, val_labels) if p == t) / len(val_labels)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print("-"*30)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"           | Train Acc:  {train_acc:.4f}, Val Acc:  {val_acc:.4f}")
    print("-"*30)

stop = timeit.default_timer()
print(f"Total Training Time: {stop - start:.2f}s")

# =============================================================================
# Plot Loss and Accuracy Curves
# =============================================================================
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(14, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss", marker="o")
plt.plot(epochs_range, val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(epochs_range, val_accuracies, label="Validation Accuracy", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# Test Evaluation & Plot Predictions
# =============================================================================
model.eval()
pred_labels = []
pred_imgs = []
pred_ids = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Test Evaluation", position=0, leave=True):
        imgs = batch["image"].float().to(device)
        # Increase index by 1 (if needed)
        pred_ids.extend([int(i) + 1 for i in batch["index"]])
        outputs = model(imgs)
        pred_imgs.extend(imgs.detach().cpu())
        pred_labels.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy())

# Plot a grid of 3x3 test images with predicted labels
f, axarr = plt.subplots(3, 3, figsize=(10, 10))
counter = 0
for i in range(3):
    for j in range(3):
        axarr[i][j].imshow(pred_imgs[counter].squeeze(), cmap="gray")
        axarr[i][j].set_title(f"Predicted: {pred_labels[counter]}")
        counter += 1

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()