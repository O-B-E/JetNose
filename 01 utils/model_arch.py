"""
Neural Network Architecture and Inference Helper Utilities

Defines the neural network model architectures alongside
helper functions for loading model checkpoints and performing inference
on new sensor data.
"""

import numpy as np
import torch
import torch.nn as nn

# Canonical sensor feature columns, in the order models expect them.
FEATURE_COLS = ['Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V']

# Defaults for the time-series (windowed) architecture. A window is this many
# consecutive readings; the stride controls overlap when slicing training data.
DEFAULT_WINDOW_LENGTH = 15
DEFAULT_WINDOW_STRIDE = 5


class ScentCNNsmall(nn.Module):
    """
    Small 1D Convolutional Neural Network for scent classification.

    A lightweight convolutional feature extractor (two Conv1d stages with batch
    normalization) followed by global average pooling and a linear classifier
    head. Length-agnostic via AdaptiveAvgPool1d, so it accepts any input length;
    inputs shaped (batch, num_features) are treated as a single-channel sequence.
    """

    def __init__(self, num_classes=5, in_channels=1, dropout_rate=0.25):
        """
        Initialize the ScentCNNsmall model.

        Parameters:
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
    Multivariate time-series 1D CNN for scent classification.

    Unlike the single-sample models, this network operates on a *window* of
    consecutive readings. The six sensors are placed on the channel axis and
    time is placed on the length axis, so the input is shaped
    (batch, in_channels=6, window_length). With this layout, convolution is
    justified: it slides shared kernels along the real (ordered) time axis to
    capture temporal dynamics, and because every Conv1d filter spans all input
    channels, it learns cross-sensor combinations at every time step. This
    directly exploits the inter-sensor relationships that the sensor-as-sequence
    models could not.

    Architecture: a convolutional stem, three convolutional blocks that widen
    the channels (6 -> 32 -> 64 -> 128 -> 128) while the two middle blocks
    downsample time via stride-2 max pooling, then global average pooling over
    time and a two-layer classifier head. Global pooling keeps the network
    agnostic to the exact window length.
    """

    # Marks this architecture as consuming windows shaped (batch, sensors, time)
    # rather than single rows. Data/inference code branches on this flag.
    time_series = True

    def __init__(self, num_classes=5, in_channels=6, dropout_rate=0.25):
        """
        Initialize the ScentCNNrobust model.

        Parameters:
            num_classes (int): Number of target output classes (default 5)
            in_channels (int): Number of input sensor channels; one per sensor
                (default 6: Temp, Hum, NO2_V, Eth_V, VOC_V, CO_V)
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
        Perform forward pass through the time-series 1D CNN.

        Parameters:
            x (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, window_length), i.e. sensors on the
                channel axis and time on the length axis.

        Returns:
            torch.Tensor: Unnormalized class logits output by the network
        """
        if x.dim() != 3:
            raise ValueError(
                'ScentCNNrobust expects a 3D input of shape '
                '(batch, in_channels, window_length); got shape '
                f'{tuple(x.shape)}. This model consumes windows of consecutive '
                'readings with sensors on the channel axis, not single rows.'
            )

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


def make_windows(X, y=None, window_length=DEFAULT_WINDOW_LENGTH, stride=DEFAULT_WINDOW_STRIDE):
    """
    Slice a (rows, features) array into sliding windows for the time-series model.

    Each window is transposed to (features, window_length) so that features
    (sensors) become the channel axis and time becomes the length axis, matching
    ScentCNNrobust's expected input.

    Parameters:
        X (array-like): Scaled feature matrix of shape (num_rows, num_features).
        y (array-like, optional): Per-row integer labels of shape (num_rows,). When
            provided, windows are cut only within consecutive same-label segments
            (never spanning a label change), and each window inherits its
            segment's label.
        window_length (int): Number of consecutive readings per window.
        stride (int): Step between successive window start indices (stride <
            window_length yields overlapping windows).

    Returns:
        numpy.ndarray: Windows of shape (num_windows, num_features, window_length)
            when y is None.
        tuple(numpy.ndarray, numpy.ndarray): (windows, window_labels) when y is
            provided.

    Notes:
        Segments shorter than window_length produce no windows and are skipped.
    """
    X = np.asarray(X, dtype=np.float32)

    def _slice(block):
        # block: (rows, features) -> list of (features, window_length)
        out = []
        for start in range(0, len(block) - window_length + 1, stride):
            out.append(block[start:start + window_length].T)  # (features, W)
        return out

    if y is None:
        windows = _slice(X)
        if not windows:
            raise ValueError(
                f'Cannot form a window of length {window_length} from only '
                f'{len(X)} rows. Provide at least {window_length} consecutive '
                'readings.'
            )
        return np.stack(windows)

    y = np.asarray(y)
    # Consecutive same-label segment ids (so a window never mixes two labels).
    seg_ids = np.concatenate([[0], np.cumsum(y[1:] != y[:-1])]) if len(y) else np.array([], dtype=int)

    win_list, lab_list = [], []
    for seg in np.unique(seg_ids):
        idx = np.where(seg_ids == seg)[0]
        seg_windows = _slice(X[idx])
        win_list.extend(seg_windows)
        lab_list.extend([y[idx[0]]] * len(seg_windows))

    if not win_list:
        raise ValueError(
            f'No windows of length {window_length} could be formed; every '
            'same-label segment is shorter than the window. Reduce '
            'window_length or collect longer runs per label.'
        )
    return np.stack(win_list), np.asarray(lab_list)


def load_model(path='model/CNNsmall.pth', device=None):
    """
    Load saved PyTorch model checkpoint along with feature scaler and label encoder.

    Parameters:
        path (str): Path to saved PyTorch checkpoint file (default 'model/CNNsmall.pth')
        device (torch.device, optional): Hardware device to run model on (CPU or CUDA)

    Returns:
        tuple: (model, arch_name, scaler, label_encoder) where model is set to
            evaluation mode. For time-series architectures the returned model also
            carries a ``window_length`` attribute (the window size to feed it).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    # Backward compatible: older checkpoints have no 'arch' key -> ScentCNNsmall.
    arch_name = checkpoint.get('arch', 'ScentCNNsmall')
    model_cls = ARCHITECTURES.get(arch_name, ScentCNNsmall)

    if getattr(model_cls, 'time_series', False):
        # Windowed model: input channels are the sensors, and the length axis is
        # the window of time steps. num_features carries the window length.
        window_length = checkpoint.get('window_length', checkpoint.get('num_features'))
        in_channels = checkpoint.get('in_channels', len(FEATURE_COLS))
        model = model_cls(
            num_features=window_length,
            num_classes=checkpoint['num_classes'],
            in_channels=in_channels,
        ).to(device)
        model.window_length = window_length
        model.window_stride = checkpoint.get('window_stride', DEFAULT_WINDOW_STRIDE)
    else:
        model = model_cls(
            num_features=checkpoint['num_features'],
            num_classes=checkpoint['num_classes'],
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = checkpoint['scaler']
    label_encoder = checkpoint['label_encoder']

    extra = ''
    if getattr(model, 'time_series', False):
        extra = f', window_length={getattr(model, "window_length", "?")}'
    print(f'Model loaded (arch={arch_name}{extra}).')
    return model, arch_name, scaler, label_encoder


def _scale_features(scaler, new_data_df):
    """
    Extract and scale the sensor feature columns from a DataFrame.

    Pads with zero columns when the scaler was fit on more features than are
    present (legacy padded checkpoints), then applies the scaler.

    Returns:
        numpy.ndarray: Scaled feature matrix of shape (num_rows, expected_features).
    """
    features = new_data_df[FEATURE_COLS].copy()
    expected = getattr(scaler, 'n_features_in_', len(FEATURE_COLS))
    while features.shape[1] < expected:
        features[f'pad_{features.shape[1]}'] = 0.0
    return scaler.transform(features.iloc[:, :expected])


def predict(model, scaler, label_encoder, new_data_df, device=None,
            window_length=None, stride=None):
    """
    Run model inference on a DataFrame of sensor readings.

    Dispatches on the model type:

    - Single-sample models (MLPs, ScentCNNsmall): each row is classified
      independently, returning one predicted label per row.
    - Time-series model (ScentCNNrobust): the rows are sliced into windows of
      shape (sensors, window_length) and each window is classified, returning one
      predicted label per window.

    Parameters:
        model (nn.Module): Trained PyTorch model instance
        scaler (StandardScaler): Scaler instance used for feature normalization
        label_encoder (LabelEncoder): Encoder mapping numeric predictions back to class labels
        new_data_df (pandas.DataFrame): DataFrame containing raw sensor measurements
        device (torch.device, optional): Hardware device to run inference on (CPU or CUDA)
        window_length (int, optional): Window size for the time-series model;
            defaults to the model's stored ``window_length``. Ignored otherwise.
        stride (int, optional): Window stride for the time-series model; defaults
            to the model's stored ``window_stride`` (or non-overlapping). Ignored
            otherwise.

    Returns:
        numpy.ndarray: Array of predicted class label names (one per row for
            single-sample models, one per window for the time-series model).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_new = _scale_features(scaler, new_data_df)

    if getattr(model, 'time_series', False):
        w = window_length or getattr(model, 'window_length', DEFAULT_WINDOW_LENGTH)
        s = stride or w  # default: non-overlapping windows (one label per window)
        windows = make_windows(X_new, window_length=w, stride=s)  # (N, sensors, W)
        inputs = torch.FloatTensor(windows).to(device)
    else:
        inputs = torch.FloatTensor(X_new).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    return label_encoder.inverse_transform(predicted.cpu().numpy())
