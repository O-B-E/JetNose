"""
Neural Network Architecture and Inference Helper Utilities

Defines the NNetwork PyTorch neural network model architecture alongside
helper functions for loading model checkpoints and performing inference
on new sensor data.
"""

import torch
import torch.nn as nn


class NNetwork(nn.Module):
    """
    Neural network model architecture for scent classification.

    Consists of input, hidden, and output fully-connected layers with
    ReLU activations and dropout for regularization.
    """

    def __init__(self, num_features=40, num_classes=5, dropout_rate=0.25):
        """
        Initialize the NNetwork model.

        Parameters:
            num_features (int): Number of input features (default 40)
            num_classes (int): Number of target output classes (default 5)
            dropout_rate (float): Probability of dropping out nodes during training (default 0.25)
        """
        super(NNetwork, self).__init__()
        self.input_layer = nn.Linear(num_features, 40)
        self.hidden_layer = nn.Linear(40, 20)
        self.output_layer = nn.Linear(20, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Perform forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor containing feature data

        Returns:
            torch.Tensor: Unnormalized class logits output by the network
        """
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def load_model(path='model/scent_classifier.pth', device=None):
    """
    Load saved PyTorch model checkpoint along with feature scaler and label encoder.

    Parameters:
        path (str): Path to saved PyTorch checkpoint file (default 'model/scent_classifier.pth')
        device (torch.device, optional): Hardware device to run model on (CPU or CUDA)

    Returns:
        tuple: (model, scaler, label_encoder) where model is set to evaluation mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)
    model = NNetwork(
        num_features=checkpoint['num_features'],
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = checkpoint['scaler']
    label_encoder = checkpoint['label_encoder']

    print('Model loaded.')
    return model, scaler, label_encoder


def predict(model, scaler, label_encoder, new_data_df, device=None):
    """
    Run model inference on a DataFrame of sensor readings.

    Extracts required feature columns, pads feature vector to 40 inputs,
    scales data, and runs forward inference to return predicted label strings.

    Parameters:
        model (NNetwork): Trained PyTorch model instance
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

    # Pad to 40 features
    while features.shape[1] < 40:
        features[f'pad_{features.shape[1]}'] = 0.0

    X_new = scaler.transform(features.iloc[:, :40])
    inputs = torch.FloatTensor(X_new).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    return label_encoder.inverse_transform(predicted.cpu().numpy())