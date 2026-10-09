"""
Scent Classification Model Training Pipeline

Provides functions for loading sensor data from CSV files, performing feature
preprocessing and scaling, training the PyTorch neural network classifier,
and saving complete checkpoints containing model weights, scaling parameters,
and label encoders.
"""

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
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
    from model_arch import (
        ScentCNNsmall, ScentCNNrobust, make_windows, assign_groups, FEATURE_COLS,
        DEFAULT_WINDOW_LENGTH, DEFAULT_WINDOW_STRIDE,
    )
    ARCH_MAP = {'small': ScentCNNsmall, 'robust': ScentCNNrobust}
except ImportError as e:
    print("\nError: Failed to import Neural Network architecture.")
    print(f"Details: {e}")
    sys.exit(1)


def _windows_per_run(X, y, groups, run_ids, window_length, window_stride):
    """
    Slice each recording run into windows separately, so no window spans two
    runs (a label change or a time gap) and every window belongs to one split.

    Runs shorter than window_length yield no windows and are skipped.

    Returns:
        tuple: (windows, labels) of shapes (N, num_sensors, window_length) and (N,)
    """
    win_list, lab_list = [], []
    for run in run_ids:
        rows = np.flatnonzero(groups == run)
        if len(rows) < window_length:
            continue
        w, lab = make_windows(X[rows], y[rows], window_length=window_length, stride=window_stride)
        win_list.append(w)
        lab_list.append(lab)
    if not win_list:
        raise ValueError(
            f'No windows of length {window_length} could be formed in this split; '
            'every run is shorter than the window.'
        )
    return np.concatenate(win_list), np.concatenate(lab_list)


def prepare_data(csv_file, val_split=0.2, time_series=False,
                 window_length=DEFAULT_WINDOW_LENGTH, window_stride=DEFAULT_WINDOW_STRIDE,
                 gap_seconds=2.0, random_state=42):
    """
    Load the dataset and produce a leakage-safe train/val split (80/20 by default).

    Rows are grouped into recording runs (new run on a label change or a time gap
    above gap_seconds) and whole runs are assigned to train or val, so adjacent
    near-duplicate readings never straddle the split. The scaler is fit on the
    training rows only.

    For ScentCNNsmall each row is one sample of shape (num_features,). For the
    time-series ScentCNNrobust each run is sliced into windows of shape
    (num_sensors, window_length) after the split, so overlapping windows
    (stride < window_length) stay within one split.

    Parameters:
        csv_file (str): Path to CSV dataset file containing raw sensor telemetry
        val_split (float): Fraction of runs reserved for validation (default 0.2)
        time_series (bool): When True, produce windowed samples for ScentCNNrobust
        window_length (int): Window size (time steps) when time_series is True
        window_stride (int): Window stride when time_series is True
        gap_seconds (float): Time gap that separates recording runs
        random_state (int): Seed for the group splitter

    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler, label_encoder)
    """
    df = pd.read_csv(csv_file)
    features = df[FEATURE_COLS]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Label'])
    groups = assign_groups(df, gap_seconds=gap_seconds)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)
    train_idx, val_idx = next(gss.split(features, y, groups))

    # Fit on training rows only, then apply the same transform to every row.
    scaler = StandardScaler()
    scaler.fit(features.iloc[train_idx])
    X = scaler.transform(features).astype(np.float32)

    if time_series:
        X_train, y_train = _windows_per_run(
            X, y, groups, np.unique(groups[train_idx]), window_length, window_stride)
        X_val, y_val = _windows_per_run(
            X, y, groups, np.unique(groups[val_idx]), window_length, window_stride)
        return X_train, X_val, y_train, y_val, scaler, label_encoder

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx], scaler, label_encoder


def train_model(model_name, csv_file='csv/training_data.csv', num_epochs=300, lr=0.0005, weight_decay=1e-4, val_split=0.2, batch_size=32, history_dir='logs', arch='small', patience=10, gap_seconds=2.0, window_length=DEFAULT_WINDOW_LENGTH, window_stride=DEFAULT_WINDOW_STRIDE):
    """
    Train a scent classifier with a grouped 80/20 split, class-weighted loss,
    weight decay, LR scheduling, early stopping, and per-epoch history logging.

    Parameters:
        model_name (str): Model name used for saving the checkpoint and history CSVs
        csv_file (str): Path to CSV training dataset file (default 'csv/training_data.csv')
        num_epochs (int): Maximum number of training epochs (default 300)
        lr (float): Initial learning rate for Adam optimizer (default 0.0005)
        weight_decay (float): L2 regularization strength (default 1e-4)
        val_split (float): Fraction of recording runs reserved for validation (default 0.2)
        batch_size (int): Mini-batch size for DataLoaders (default 32)
        history_dir (str): Directory (relative to this script) for per-epoch history CSVs (default 'logs')
        arch (str): Model architecture to use (default 'small')
        patience (int): Epochs without val-loss improvement before early stopping (default 10)
        gap_seconds (float): Time gap that separates recording runs (default 2.0)
        window_length (int): Window size (time steps) for time-series architectures
        window_stride (int): Window stride for time-series architectures

    Returns:
        tuple: (model, scaler, label_encoder) trained PyTorch model and preprocessing artifacts
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_cls = ARCH_MAP.get(arch, ScentCNNsmall)
    time_series = getattr(model_cls, 'time_series', False)

    # Prepare data
    X_train, X_val, y_train, y_val, scaler, label_encoder = prepare_data(
        csv_file, val_split, time_series=time_series,
        window_length=window_length, window_stride=window_stride,
        gap_seconds=gap_seconds,
    )

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
    if time_series:
        # X_train shape: (num_windows, num_sensors, window_length). The CNN is
        # length-agnostic (global pooling), so it only needs in_channels.
        in_channels = X_train.shape[1]
        win_len = X_train.shape[2]
        model = model_cls(num_classes=num_classes, in_channels=in_channels).to(device)
        # Record the window contract so save_model/load_model can round-trip it.
        model.window_length = win_len
        model.window_stride = window_stride
        num_features = win_len
    else:
        # Row-based CNN (ScentCNNsmall): single-channel input, length-agnostic.
        num_features = X_train.shape[1]
        model = model_cls(num_classes=num_classes).to(device)

    # Class weighting to counter label imbalance.
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = torch.FloatTensor(
        len(y_train) / (num_classes * np.clip(class_counts, 1.0, None))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = []
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0

    # Training loop
    if time_series:
        print(f'Started Training (arch={arch}, time-series, sensors={in_channels}, '
              f'window_length={num_features}, stride={window_stride}, classes={num_classes})')
    else:
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
    summary = (f'Training completed. Restored best model from epoch {best_epoch} (val loss {best_val_loss:.4f}).')
    print(summary)

    # Persist per-epoch history.
    hist_dir = os.path.join(current_dir, history_dir)
    os.makedirs(hist_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hist_path = os.path.join(hist_dir, f'training_history_{model_name}_{stamp}.csv')
    pd.DataFrame(history).to_csv(hist_path, index=False)
    with open(hist_path, 'a') as f:
        f.write(f'# {summary}\n')
        f.write(f'\n# Training file: {os.path.normpath(csv_file)}\n')
        if time_series:
            f.write(f'# Window length: {num_features}, stride: {window_stride}\n')
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

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'arch': model.__class__.__name__,
        'num_features': int(scaler.n_features_in_),
        'num_classes': len(label_encoder.classes_),
    }

    # Time-series models need their window contract to rebuild input correctly:
    # num_features carries the window length, plus the sensor channel count.
    if getattr(model, 'time_series', False):
        window_length = int(getattr(model, 'window_length', DEFAULT_WINDOW_LENGTH))
        checkpoint['num_features'] = window_length
        checkpoint['window_length'] = window_length
        checkpoint['window_stride'] = int(getattr(model, 'window_stride', DEFAULT_WINDOW_STRIDE))
        checkpoint['in_channels'] = int(getattr(scaler, 'n_features_in_', len(FEATURE_COLS)))

    torch.save(checkpoint, path)
    print(f'Model saved to {path}')


# Usage example with interactive prompts:
if __name__ == "__main__":
    while True:
        csv_name = input("Enter CSV file name (default: training_data): ") or 'training_data'
        csv_path = os.path.join('csv', f'{csv_name}.csv')
        if os.path.exists(csv_path):
            break
        print(f'File not found: {csv_path}. Please try again.')
    print(f'Using CSV file: {csv_path}')

    arch = (input("Choose architecture configuration [small/robust] (default: robust): ") or 'robust').strip().lower()

    # The 'robust' CNN is a time-series model: prompt for the window contract.
    window_length = DEFAULT_WINDOW_LENGTH
    window_stride = DEFAULT_WINDOW_STRIDE
    if getattr(ARCH_MAP.get(arch, ScentCNNsmall), 'time_series', False):
        window_length = int(input(f"Window length in samples (default: {DEFAULT_WINDOW_LENGTH}): ") or DEFAULT_WINDOW_LENGTH)
        window_stride = int(input(f"Window stride in samples (default: {DEFAULT_WINDOW_STRIDE}): ") or DEFAULT_WINDOW_STRIDE)

    model_name = input("Enter model name (default: CNN_model): ") or 'CNN_model'
    model_path = os.path.join('model', f'{model_name}.pth')
    print(f'Model architecture configuration: {arch}, will be saved to: {model_path}')

    model, scaler, label_encoder = train_model(
        model_name=model_name, csv_file=csv_path, arch=arch,
        window_length=window_length, window_stride=window_stride,
    )
    save_model(model, scaler, label_encoder, path=model_path)


