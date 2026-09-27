"""
Neural Network Architecture and Inference Helper Utilities

Defines the neural network model architectures alongside
helper functions for loading model checkpoints and performing inference
on new sensor data.
"""

import torch
import torch.nn as nn


class ScentCNNsmall(nn.Module):
    """
    Small 1D Convolutional Neural Network for scent classification.

    A lightweight convolutional feature extractor (two Conv1d stages with batch
    normalization) followed by global average pooling and a linear classifier
    head. Length-agnostic via AdaptiveAvgPool1d, so it accepts any input length;
    inputs shaped (batch, num_features) are treated as a single-channel sequence.
    """

    def __init__(self, num_features=6, num_classes=5, in_channels=1, dropout_rate=0.25):
        """
        Initialize the ScentCNNsmall model.

        Parameters:
            num_features (int): Input sequence length (default 6). Informational
                only; the network is length-agnostic thanks to global pooling.
            num_classes (int): Number of target output classes (default 5)
            in_channels (int): Number of input sensor channels (default 1)
            dropout_rate (float): Probability of dropping out nodes during training (default 0.25)
        """
        super(ScentCNNsmall, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        Perform forward pass through the 1D CNN.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features) or
                (batch_size, channels, sequence_length)

        Returns:
            torch.Tensor: Unnormalized class logits output by the network
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)                       # (N, 1, L)

        x = self.relu(self.bn1(self.conv1(x)))       # (N, 16, L)
        x = self.relu(self.bn2(self.conv2(x)))       # (N, 32, L)
        x = self.dropout(x)
        x = self.global_pool(x).squeeze(-1)          # (N, 32)
        x = self.classifier(x)                       # (N, num_classes)
        return x


class ScentCNNrobust(nn.Module):
    """
    1D Convolutional Neural Network architecture for scent classification.

    Optimized for edge deployment on NVIDIA Jetson Nano. Uses multi-stage 1D
    convolutions, batch normalization, max pooling, and global average pooling
    to extract local feature dynamics from multi-sensor telemetry.
    """

    def __init__(self, num_features=6, num_classes=5, in_channels=1, dropout_rate=0.25):
        """
        Initialize the ScentCNNrobust model.

        Parameters:
            num_features (int): Number of input features/sequence length (default 40)
            num_classes (int): Number of target output classes (default 5)
            in_channels (int): Number of input sensor channels (default 1)
            dropout_rate (float): Probability of dropping out nodes during training (default 0.25)
        """
        super(ScentCNNrobust, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Perform forward pass through the 1D CNN network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features) or (batch_size, channels, sequence_length)

        Returns:
            torch.Tensor: Unnormalized class logits output by the network
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x



class ScentMLPSmall(nn.Module):
    """
    Small fully-connected classifier for tabular sensor features.

    Sized to the actual number of input features (no zero-padding). Intended as
    the lightweight architecture for edge deployment where model size matters.
    """

    def __init__(self, num_features=6, num_classes=5, dropout_rate=0.3):
        """
        Initialize the ScentMLPSmall model.

        Parameters:
            num_features (int): Number of real input features (default 6)
            num_classes (int): Number of target output classes (default 5)
            dropout_rate (float): Dropout probability during training (default 0.3)
        """
        super(ScentMLPSmall, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        """Perform forward pass, returning unnormalized class logits."""
        return self.net(x)


class ScentMLPRobust(nn.Module):
    """
    Wider, strongly-regularized fully-connected classifier for tabular features.

    Uses BatchNorm + Dropout in each hidden block to provide regularized capacity
    without the false locality assumption a 1D CNN would impose on unordered
    sensor channels. See plan.md for the MLP-vs-CNN rationale.
    """

    def __init__(self, num_features=6, num_classes=5, dropout_rate=0.4):
        """
        Initialize the ScentMLPRobust model.

        Parameters:
            num_features (int): Number of real input features (default 6)
            num_classes (int): Number of target output classes (default 5)
            dropout_rate (float): Dropout probability during training (default 0.4)
        """
        super(ScentMLPRobust, self).__init__()

        def block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

        self.features = nn.Sequential(
            block(num_features, 128),
            block(128, 64),
            block(64, 32)
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        """Perform forward pass, returning unnormalized class logits."""
        x = self.features(x)
        x = self.classifier(x)
        return x


# Registry mapping checkpoint architecture names to their classes.
ARCHITECTURES = {
    'ScentCNNsmall': ScentCNNsmall,
    'ScentCNNrobust': ScentCNNrobust,
    'ScentMLPSmall': ScentMLPSmall,
    'ScentMLPRobust': ScentMLPRobust,
}


def load_model(path='model/CNNsmall.pth', device=None):
    """
    Load saved PyTorch model checkpoint along with feature scaler and label encoder.

    Parameters:
        path (str): Path to saved PyTorch checkpoint file (default 'model/CNNsmall.pth')
        device (torch.device, optional): Hardware device to run model on (CPU or CUDA)

    Returns:
        tuple: (model, scaler, label_encoder) where model is set to evaluation mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    # Backward compatible: older checkpoints have no 'arch' key -> ScentCNNsmall.
    arch_name = checkpoint.get('arch', 'ScentCNNsmall')
    model_cls = ARCHITECTURES.get(arch_name, ScentCNNsmall)
    model = model_cls(
        num_features=checkpoint['num_features'],
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = checkpoint['scaler']
    label_encoder = checkpoint['label_encoder']

    print(f'Model loaded (arch={arch_name}).')
    return model, arch_name, scaler, label_encoder


def predict(model, scaler, label_encoder, new_data_df, device=None):
    """
    Run model inference on a DataFrame of sensor readings.

    Extracts required feature columns, pads feature vector to 40 inputs,
    scales data, and runs forward inference to return predicted label strings.

    Parameters:
        model (ScentCNNsmall): Trained PyTorch model instance
        scaler (StandardScaler): Scaler instance used for feature normalization
        label_encoder (LabelEncoder): Encoder mapping numeric predictions back to class labels
        new_data_df (pandas.DataFrame): DataFrame containing raw sensor measurements
        device (torch.device, optional): Hardware device to run inference on (CPU or CUDA)

    Returns:
        numpy.ndarray: Array of predicted class label names
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_cols = ['Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V']
    features = new_data_df[feature_cols].copy()

    # Match the number of features the scaler was fit on. New models are trained
    # on the real feature count (no padding); legacy models were padded to 40.
    expected = getattr(scaler, 'n_features_in_', len(feature_cols))
    while features.shape[1] < expected:
        features[f'pad_{features.shape[1]}'] = 0.0

    X_new = scaler.transform(features.iloc[:, :expected])
    inputs = torch.FloatTensor(X_new).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    return label_encoder.inverse_transform(predicted.cpu().numpy())
