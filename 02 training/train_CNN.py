"""
Scent Classification Model Training Pipeline

Provides functions for loading sensor data from CSV files, performing feature
preprocessing and scaling, training the PyTorch neural network classifier,
and saving complete checkpoints containing model weights, scaling parameters,
and label encoders.
"""

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import copy
import os
import sys
from datetime import datetime

# Importing utils
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../01 utils')
sys.path.append(utils_path)

try:
    from model_arch import ScentCNNsmall, ScentCNNrobust
    ARCH_MAP = {'small': ScentCNNsmall, 'robust': ScentCNNrobust}
except ImportError as e:
    print(f"\nError: Failed to import Neural Network architecture.")
    print(f"Details: {e}")
    sys.exit(1)


def prepare_data(csv_file, val_split=0.2):
    """
    Load sensor dataset from CSV, scale features, encode labels, and split into train/val sets.

    Parameters:
        csv_file (str): Path to CSV dataset file containing raw sensor telemetry
        val_split (float): Ratio of dataset reserved for validation (default 0.2)

    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler, label_encoder)
    """
    df = pd.read_csv(csv_file)

    feature_cols = ['Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V']
    features = df[feature_cols].copy()

    # Scale the real sensor features (no zero-padding).
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Label'])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val, scaler, label_encoder


def train_model(model_name, csv_file='csv/training_data.csv', num_epochs=300, lr=0.0005, val_split=0.2, batch_size=32, history_dir='logs', arch='small', patience=10):
    """
    Execute full training and validation loop for the scent classification model.

    Parameters:
        model_name (str): Model name used for saving the checkpoint and history CSVs
        csv_file (str): Path to CSV training dataset file (default 'csv/training_data.csv')
        num_epochs (int): Total number of training epochs (default 300)
        lr (float): Learning rate for Adam optimizer (default 0.0005)
        val_split (float): Ratio of dataset reserved for validation (default 0.2)
        batch_size (int): Mini-batch size for DataLoaders (default 32)
        history_dir (str): Directory (relative to this script) for per-epoch history CSVs (default 'logs')
        arch (str): Model architecture to use (default 'small')
        patience (int): Epochs without val-loss improvement before early stopping (default 10)

    Returns:
        tuple: (model, scaler, label_encoder) trained PyTorch model and preprocessing artifacts
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    X_train, X_val, y_train, y_val, scaler, label_encoder = prepare_data(csv_file, val_split)

    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    # drop_last avoids a size-1 final batch, which BatchNorm1d cannot handle in
    # training mode (only dropped when there is more than one full batch).
    drop_last = len(y_train) > batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_classes = len(label_encoder.classes_)
    num_features = X_train.shape[1]
    model = ARCH_MAP.get(arch, ScentCNNsmall)(num_features=num_features, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0

    # Training loop
    print(f'Started Training (arch={arch}, features={num_features}, classes={num_classes})')
    print(f'Split sizes -> train: {len(y_train)}, val: {len(y_val)}')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= max(len(val_loader), 1)
        val_acc = 100.0 * correct / max(total, 1)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Early stopping / best-checkpoint tracking.
        if val_loss < best_val_loss - 1e-4:  # Significant improvement
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]["lr"]:.2e}')

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1} '
                  f'(no val-loss improvement for {patience} epochs).')
            break

    # Restore the best weights before saving.
    model.load_state_dict(best_state)
    print(f'Training completed. Restored best model from epoch {best_epoch} '
          f'(val loss {best_val_loss:.4f}).')

    # Persist per-epoch history.
    hist_dir = os.path.join(current_dir, history_dir)
    os.makedirs(hist_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hist_path = os.path.join(hist_dir, f'training_history_{model_name}_{stamp}.csv')
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f'Training history saved to {hist_path}')

    return model, scaler, label_encoder


def save_model(model, scaler, label_encoder, path='model/scent_classifier.pth'):
    """
    Save trained model state dictionary along with scaler and label encoder to a file.

    Parameters:
        model (nn.Module): Trained PyTorch neural network model instance
        scaler (StandardScaler): Feature scaler instance used during training
        label_encoder (LabelEncoder): Label encoder instance used for target classes
        path (str): Filepath where checkpoint should be saved (default 'model/scent_classifier.pth')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'arch': model.__class__.__name__,
        'num_features': int(scaler.n_features_in_),
        'num_classes': len(label_encoder.classes_)
    }, path)
    print(f'Model saved to {path}')


# Usage example with interactive prompts:
if __name__ == "__main__":
    csv_name = input("Enter CSV file name (default: training_data): ") or 'training_data'
    csv_path = os.path.join('csv', f'{csv_name}.csv')
    print(f'Using CSV file: {csv_path}')

    arch = input("Enter model architecture (default: small): ") or 'small'
    model_name = input("Enter model name (default: CNN_model): ") or 'CNN_model'
    model_path = os.path.join('model', f'{model_name}.pth')
    print(f'Model architecture configuration: {arch}, will be saved to: {model_path}')

    model, scaler, label_encoder = train_model(model_name=model_name, csv_file=csv_path, arch=arch)
    save_model(model, scaler, label_encoder, path=model_path)


