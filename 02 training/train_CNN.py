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
    from model_arch import (
        ScentCNNsmall, ScentCNNrobust, make_windows, FEATURE_COLS,
        DEFAULT_WINDOW_LENGTH, DEFAULT_WINDOW_STRIDE,
    )
    ARCH_MAP = {'small': ScentCNNsmall, 'robust': ScentCNNrobust}
except ImportError as e:
    print(f"\nError: Failed to import Neural Network architecture.")
    print(f"Details: {e}")
    sys.exit(1)


def prepare_data(csv_file, val_split=0.2, time_series=False,
                 window_length=DEFAULT_WINDOW_LENGTH, window_stride=DEFAULT_WINDOW_STRIDE):
    """
    Load sensor dataset from CSV, scale features, encode labels, and split into train/val sets.

    For single-sample architectures each CSV row is one sample of shape
    (num_features,). For the time-series architecture the rows are sliced into
    windows of shape (num_sensors, window_length) — windows never span a label
    change — and each window is one sample.

    Parameters:
        csv_file (str): Path to CSV dataset file containing raw sensor telemetry
        val_split (float): Ratio of dataset reserved for validation (default 0.2)
        time_series (bool): When True, produce windowed samples for ScentCNNrobust
        window_length (int): Window size (time steps) when time_series is True
        window_stride (int): Window stride when time_series is True (stride <
            window_length yields overlapping windows for more training samples)

    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler, label_encoder)
    """
    df = pd.read_csv(csv_file)
    features = df[FEATURE_COLS].copy()

    # Scale the real sensor features (no zero-padding).
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Label'])

    if time_series:
        # Slice the scaled per-row stream into (sensors, time) windows, cut within
        # consecutive same-label segments only. NOTE: overlapping windows
        # (window_stride < window_length) can place near-identical windows on both
        # sides of the split; set window_stride == window_length for a clean,
        # leakage-free split at the cost of fewer samples.
        Xw, yw = make_windows(X, y, window_length=window_length, stride=window_stride)
        X_train, X_val, y_train, y_val = train_test_split(
            Xw, yw, test_size=val_split, random_state=42, stratify=yw
        )
        return X_train, X_val, y_train, y_val, scaler, label_encoder

    # Split data (row-wise samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val, scaler, label_encoder


def train_model(model_name, csv_file='csv/training_data.csv', num_epochs=300, lr=0.0005, val_split=0.2, batch_size=32, history_dir='logs', arch='small', patience=10, window_length=DEFAULT_WINDOW_LENGTH, window_stride=DEFAULT_WINDOW_STRIDE):
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
        # X_train shape: (num_windows, num_sensors, window_length)
        in_channels = X_train.shape[1]
        win_len = X_train.shape[2]
        model = model_cls(num_features=win_len, num_classes=num_classes, in_channels=in_channels).to(device)
        # Record the window contract so save_model/load_model can round-trip it.
        model.window_length = win_len
        model.window_stride = window_stride
        num_features = win_len
    else:
        num_features = X_train.shape[1]
        model = model_cls(num_features=num_features, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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


