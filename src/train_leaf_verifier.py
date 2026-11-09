"""
Training Script for Dedicated Binary Leaf Verifier CNN Model.
Classifies images into:
  - Class 0: non_leaf (human faces, animals, vehicles, buildings, fruits, objects, etc.)
  - Class 1: leaf (genuine plant leaves)
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "leaf_verifier")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

def build_leaf_verifier_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze earlier layers
    for param in model.features[:12].parameters():
        param.requires_grad = False
    for param in model.features[12:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(128, 2)  # [non_leaf, leaf]
    )
    return model

def get_data_loaders(batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names

def train_verifier(epochs=6, batch_size=32, lr=0.0008):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Starting Leaf Verifier CNN Training on {device}...")
    print(f"=======================================================")

    train_loader, val_loader, test_loader, class_names = get_data_loaders(batch_size=batch_size)
    print(f"Verifier Classes ({len(class_names)}): {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Save classes map
    with open(os.path.join(MODELS_DIR, "verifier_classes.json"), "w") as f:
        json.dump({
            "class_to_idx": {name: i for i, name in enumerate(class_names)},
            "idx_to_class": {i: name for i, name in enumerate(class_names)}
        }, f, indent=2)

    model = build_leaf_verifier_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "leaf_verifier_cnn.pth")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # Train
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Val
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": epoch_val_acc,
                "class_names": class_names,
                "architecture": "MobileNetV2_BinaryVerifier"
            }, best_model_path)
            print(f"  --> Best verifier model saved (Val Acc: {best_val_acc*100:.2f}%)")

    total_training_time = time.time() - start_time
    print(f"\nLeaf Verifier training finished in {total_training_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    plot_training_history(history, os.path.join(RESULTS_DIR, "leaf_verifier_training_history.png"))
    evaluate_verifier_test_set(model, best_model_path, test_loader, class_names, device)

def plot_training_history(history, save_path):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-o', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, [a * 100 for a in history["val_acc"]], 'g-s', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Leaf Verifier CNN - Accuracy Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.plot(epochs, history["train_loss"], 'r-o', label='Train Loss', linewidth=2)
    ax2.plot(epochs, history["val_loss"], 'm-s', label='Validation Loss', linewidth=2)
    ax2.set_title('Leaf Verifier CNN - Loss Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss (CrossEntropy)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Verifier history plot saved to: {save_path}")

def evaluate_verifier_test_set(model, best_model_path, test_loader, class_names, device):
    print(f"\n=======================================================")
    print(f"Evaluating Leaf Verifier on Unseen Test Set...")
    print(f"=======================================================")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nVerifier Classification Report:\n" + report_text)

    with open(os.path.join(RESULTS_DIR, "leaf_verifier_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=class_names, yticklabels=class_names,
           title='Leaf Verifier CNN - Confusion Matrix',
           ylabel='True Class', xlabel='Predicted Class')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold', fontsize=12)

    fig.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "leaf_verifier_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Verifier confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    train_verifier(epochs=6, batch_size=32, lr=0.0008)
