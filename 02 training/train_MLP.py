"""
Scent Classification Training Pipeline

Two architectures are selectable:
  * 'small'  -> ScentMLPSmall  (lightweight)
  * 'robust' -> ScentMLPRobust (wide, BatchNorm + Dropout regularized)
"""

import copy
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

# Importing utils
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../01 utils')
sys.path.append(utils_path)

try:
    from model_arch import ScentMLPSmall, ScentMLPRobust
except ImportError as e:
    print("\nError: Failed to import Neural Network architectures.")
    print(f"Details: {e}")
    sys.exit(1)

FEATURE_COLS = ['Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V']
ARCH_MAP = {'small': ScentMLPSmall, 'robust': ScentMLPRobust}


def _assign_groups(df, gap_seconds=2.0):
    """
    Assign a group id to each row so that leakage-prone, adjacent time-series
    readings stay together in a single split.

    A new group starts whenever the Label changes or the time gap between two
    consecutive rows exceeds ``gap_seconds`` (i.e. a new recording run).

    Parameters:
        df (pandas.DataFrame): Dataset containing 'Timestamp' and 'Label' columns.
        gap_seconds (float): Time gap (seconds) that marks a new recording run.

    Returns:
        numpy.ndarray: Integer group id per row.
    """
    if 'Timestamp' in df.columns:
        ts = pd.to_datetime(df['Timestamp'], errors='coerce')
        gap = ts.diff().dt.total_seconds().fillna(0.0)
        new_run = gap > gap_seconds
    else:
        # No timestamps available: fall back to label-change boundaries only.
        new_run = pd.Series(False, index=df.index)

    label_change = df['Label'].ne(df['Label'].shift()).fillna(True)
    boundaries = (new_run | label_change).astype(int)
    return boundaries.cumsum().to_numpy()


def prepare_data(csv_file, val_split=0.2, gap_seconds=2.0, random_state=42):
    """
    Load the dataset and produce a leakage-safe train/val split (80/20 by default).

    Features are NOT zero-padded. The scaler is fit on the training split only.
    Splitting is grouped (by recording run) so adjacent near-duplicate readings
    never straddle the two splits.

    Parameters:
        csv_file (str): Path to the CSV dataset.
        val_split (float): Fraction of the dataset used for validation.
        gap_seconds (float): Time gap that separates recording runs (for grouping).
        random_state (int): Seed for the group splitter.

    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler, label_encoder, feature_cols)
    """
    df = pd.read_csv(csv_file)

    features = df[FEATURE_COLS].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Label'])
    groups = _assign_groups(df, gap_seconds=gap_seconds)

    X_all = features.to_numpy(dtype=np.float32)

    # Grouped train/val split so adjacent near-duplicate readings never straddle splits.
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)
    train_idx, val_idx = next(gss.split(X_all, y, groups))

    X_train, y_train = X_all[train_idx], y[train_idx]
    X_val, y_val = X_all[val_idx], y[val_idx]

    # Fit scaler on the training split only, then transform both splits.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, scaler, label_encoder, FEATURE_COLS


def _evaluate(model, loader, criterion, device):
    """Compute average loss and accuracy over a DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def train_model( model_name, csv_file='csv/training_data.csv', arch='robust', num_epochs=300,
                lr=0.0005, weight_decay=1e-4, val_split=0.2,
                batch_size=32, patience=10, gap_seconds=2.0, history_dir='logs'):
    """
    Train a scent classifier with early stopping, weight decay, LR scheduling,
    and per-epoch history logging, using an 80/20 train/val split.

    Parameters:
        model_name (str): Model name used for saving the checkpoint and history CSVs.
        csv_file (str): Path to the CSV training dataset.
        arch (str): 'small' or 'robust' architecture selector.
        num_epochs (int): Maximum number of training epochs.
        lr (float): Initial Adam learning rate.
        weight_decay (float): L2 regularization strength.
        val_split (float): Validation fraction of the dataset.
        batch_size (int): Mini-batch size.
        patience (int): Epochs without val-loss improvement before stopping.
        gap_seconds (float): Time gap marking a new recording run (grouping).
        history_dir (str): Directory (relative to this script) for history CSVs.

    Returns:
        tuple: (model, scaler, label_encoder, feature_cols, arch_name)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch not in ARCH_MAP:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(ARCH_MAP)}.")

    (X_train, X_val, y_train, y_val,
     scaler, label_encoder, feature_cols) = prepare_data(
        csv_file, val_split=val_split, gap_seconds=gap_seconds)

    # drop_last avoids a size-1 final batch, which BatchNorm1d cannot handle in
    # training mode (only dropped when there is more than one full batch).
    drop_last = len(y_train) > batch_size
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=batch_size, shuffle=False)

    num_classes = len(label_encoder.classes_)
    num_features = len(feature_cols)
    model_cls = ARCH_MAP[arch]
    model = model_cls(num_features=num_features, num_classes=num_classes).to(device)
    arch_name = model_cls.__name__

    # Class weighting to counter label imbalance.
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = torch.FloatTensor(
        len(y_train) / (num_classes * np.clip(class_counts, 1.0, None))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    print(f'Started Training (arch={arch_name}, features={num_features}, classes={num_classes})')
    print(f'Split sizes -> train: {len(y_train)}, val: {len(y_val)}')

    for epoch in range(num_epochs):
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

        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

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
    summary = f'Training completed. Restored best model from epoch {best_epoch} (val loss {best_val_loss:.4f}).'
    print(summary)

    # Persist per-epoch history.
    hist_dir = os.path.join(current_dir, history_dir)
    os.makedirs(hist_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hist_path = os.path.join(hist_dir, f'training_history_{model_name}_{stamp}.csv')
    pd.DataFrame(history).to_csv(hist_path, index=False)
    with open(hist_path, 'a') as f:
        f.write(f'# {summary}\n')
    print(f'Training history saved to {hist_path}')

    return model, scaler, label_encoder, feature_cols, arch_name


def save_model(model, scaler, label_encoder, feature_cols, arch_name,
               path='model/scent_classifier.pth'):
    """
    Save the trained model with everything needed to reconstruct it at inference,
    including the architecture name and the exact feature columns used.

    Parameters:
        model (nn.Module): Trained model instance.
        scaler (StandardScaler): Fitted feature scaler.
        label_encoder (LabelEncoder): Fitted label encoder.
        feature_cols (list): Ordered list of feature column names used.
        arch_name (str): Architecture class name (for load_model dispatch).
        path (str): Destination checkpoint path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'arch': arch_name,
        'feature_cols': feature_cols,
        'num_features': len(feature_cols),
        'num_classes': len(label_encoder.classes_),
    }, path)
    print(f'Model saved to {path}')


if __name__ == "__main__":
    while True:
        csv_name = input("Enter CSV file name (default: training_data): ") or 'training_data'
        csv_path = os.path.join('csv', f'{csv_name}.csv')
        if os.path.exists(csv_path):
            break
        print(f'File not found: {csv_path}. Please try again.')
    print(f'Using CSV file: {csv_path}')
    
    arch = (input("Choose architecture configuration [small/robust] (default: robust): ") or 'robust').strip().lower()

    model_name = input("Enter model name (default: MLP_model): ") or 'MLP_model'
    model_path = os.path.join('model', f'{model_name}.pth')
    print(f'Model architecture configuration: {arch}, will be saved to: {model_path}')

    model, scaler, label_encoder, feature_cols, arch_name = train_model( model_name=model_name, csv_file=csv_path, arch=arch)
    save_model(model, scaler, label_encoder, feature_cols, arch_name, path=model_path)
