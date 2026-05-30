'''
model_arch.py

Defines the `NNetwork` architecture and helper functions to load
and run inference from a saved checkpoint. 
'''

import torch
import torch.nn as nn

class NNetwork(nn.Module):
    def __init__(self, num_features=40, num_classes=5, dropout_rate=0.25):
        super(NNetwork, self).__init__()
        self.input_layer = nn.Linear(num_features, 40)
        self.hidden_layer = nn.Linear(40, 20)
        self.output_layer = nn.Linear(20, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x



def load_model(path='model/scent_classifier.pth', device=None):
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