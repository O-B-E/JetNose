"""
Real-Time Sensor Inference Pipeline

Collects live sliding windows of multi-sensor readings, preprocesses features,
executes classification using a trained PyTorch model, logs predictions and confidence
probabilities to CSV files, and displays output in real-time.

Note:
    Before running on Jetson Nano, ensure OpenMP is loaded:
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
"""

import os
import sys
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import pandas as pd

# import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../01 utils')
sys.path.append(utils_path)

try:
    from model_arch import NNetwork, load_model
    from sensor_collector import SensorDataCollector
except ImportError as e:
    print('Error: Failed to import utils.')
    print(f'Details: {e}')
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_window_as_df(collector, label="inference", num_samples=10):
    """
    Collect a window of consecutive sensor readings and format them as a DataFrame.

    Parameters:
        collector (SensorDataCollector): Active sensor data collector instance
        label (str): Label string assigned to recorded samples (default "inference")
        num_samples (int): Number of consecutive readings to aggregate per window (default 10)

    Returns:
        pandas.DataFrame: DataFrame containing collected sensor measurements
    """
    rows = []
    for i in range(num_samples):

        data = collector.collect_sample(label)
        rows.append(data)

        print(
            f"\rSample {i+1}/{num_samples} | "
            f"NO2:{data['NO2_V']:.2f}V Eth:{data['Eth_V']:.2f}V "
            f"VOC:{data['VOC_V']:.2f}V CO:{data['CO_V']:.2f}V",
            end=''
        )

        time.sleep(0.25)

    print()
    return pd.DataFrame(rows)


def predict_window(model, scaler, label_encoder, df_window):
    """
    Perform inference on a DataFrame window of sensor samples and aggregate prediction probabilities.

    Scales raw input features, computes softmax class probabilities for each sample in the window,
    calculates average confidence across the window, and identifies the top predicted class.

    Parameters:
        model (nn.Module): Trained PyTorch model instance
        scaler (StandardScaler): Feature scaler instance used during model training
        label_encoder (LabelEncoder): Label encoder instance mapping indices to class names
        df_window (pandas.DataFrame): DataFrame containing window sensor readings

    Returns:
        tuple: (best_class, best_conf, class_conf_dict) where:
            best_class (str): Name of predicted class with highest average confidence
            best_conf (float): Percentage confidence score for top predicted class
            class_conf_dict (dict): Dictionary mapping each class name to its mean confidence percentage
    """
    feature_cols = ['Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V']
    features = df_window[feature_cols].copy()

    # pad to 40 features, same logic as training
    while features.shape[1] < 40:
        features[f'pad_{features.shape[1]}'] = 0.0

    X = scaler.transform(features.iloc[:, :40])
    inputs = torch.FloatTensor(X).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)  # shape: [N, num_classes]

    # average probabilities over the window
    mean_probs = probs.mean(dim=0)  # [num_classes]
    conf_vals = mean_probs.cpu().numpy()

    # final class index and confidence
    best_idx = int(conf_vals.argmax())
    best_class = label_encoder.inverse_transform([best_idx])[0]
    best_conf = float(conf_vals[best_idx]) * 100.0

    # map class -> confidence%
    class_names = list(label_encoder.classes_)
    class_conf_dict = {
        cls: float(conf_vals[i]) * 100.0 for i, cls in enumerate(class_names)
    }

    return best_class, best_conf, class_conf_dict


def pretty_print_confidences(class_conf_dict, best_class):
    """
    Format and print per-class confidence percentages to the console.

    Parameters:
        class_conf_dict (dict): Dictionary mapping class names to confidence percentages
        best_class (str): Name of top predicted class to highlight with an arrow
    """
    print("\nWindow prediction (10 samples aggregated):")
    for cls, conf in class_conf_dict.items():
        mark = "<--" if cls == best_class else "   "
        print(f"{cls:15s}: {conf:6.2f}% {mark}")
    print()


def main():
    """
    Main execution loop for real-time sensor sampling, model loading, and live inference logging.
    """
    # Load model + preprocessing
    model_path = input(
        "Enter model path or press Enter for default: "
    ).strip() or os.path.join(current_dir, "../02 training/model/scent_classifier.pth")
    
    print("Loading Model...")
    
    model, scaler, label_encoder = load_model(path=model_path, device=DEVICE)

    # Init collector
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    collector = SensorDataCollector(
        csv_file=os.path.join(current_dir, f'csv/infer_log{timestamp}.csv'),
        sample_freq_hz=4,
        samples_per_label=10,
        warmup_time=1,
    )

    try:
        collector.warm_up()
        print("Entering real-time inference loop. CTRL+C to exit.\n")

        while True:
            # input("Press ENTER to capture 10 samples and infer...")

            # Optionally show purge/baseline between windows
            # collector.purge_check()

            df_window = collect_window_as_df(collector, label="infer", num_samples=10)
            best_class, best_conf, class_conf_dict = predict_window(
                model, scaler, label_encoder, df_window
            )
            collector.log_inference_result(
                df_window,
                best_class,
                best_conf,
                class_conf_dict,
            )
            pretty_print_confidences(class_conf_dict, best_class)

    except KeyboardInterrupt:
        print("\nInference stopped by user.")
    finally:
        collector.close()


if __name__ == "__main__":
    main()

