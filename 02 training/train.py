from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import sys

# Importing utils
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../01 utils')
sys.path.append(utils_path)

try:
    from model_arch import NNetwork
except ImportError as e:
    print(f"\nError: Failed to import Neural Network architecture.")
    print(f"Details: {e}")
    sys.exit(1)

def prepare_data(csv_file, val_split=0.2):
    df = pd.read_csv(csv_file)

    feature_cols = ['Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V']
    features = df[feature_cols].copy()

    # Pad to 40 features if needed
    while features.shape[1] < 40:
        features[f'pad_{features.shape[1]}'] = 0.0

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features.iloc[:, :40])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Label'])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val, scaler, label_encoder

def train_model(csv_file='csv/training_data.csv', num_epochs=300, lr=0.0005, val_split=0.2, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    X_train, X_val, y_train, y_val, scaler, label_encoder = prepare_data(csv_file, val_split)

    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_classes = len(label_encoder.classes_)
    model = NNetwork(num_features=40, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
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

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {100 * correct / total:.2f}%')

    print('Training completed.')
    return model, scaler, label_encoder

def save_model(model, scaler, label_encoder, path='model/scent_classifier.pth'):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'num_features': 40,
        'num_classes': len(label_encoder.classes_)
    }, path)
    print(f'Model saved to {path}')

# Usage example with interactive prompts:
csv_path = input("Enter CSV file path (default: csv/training_data.csv): ") or 'csv/training_data.csv'
model_path = input("Enter model **save** path (default: model/scent_classifier.pth): ") or 'model/scent_classifier.pth'

model, scaler, label_encoder = train_model(csv_file=csv_path)
save_model(model, scaler, label_encoder, path=model_path)

