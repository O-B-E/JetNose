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
import argparse
from datetime import datetime
from unittest import result
import torch
import torch.nn.functional as F
import pandas as pd

# import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../01 utils')
sys.path.append(utils_path)

try:
    from model_arch import load_model
    from sensor_collector import SensorDataCollector
except ImportError as e:
    print('Error: Failed to import utils.')
    print(f'Details: {e}')
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    """
    Parse command-line arguments for infer.py.

    Returns:
        argparse.Namespace: Parsed CLI options
    """
    parser = argparse.ArgumentParser(description="Real-Time Sensor Inference Pipeline")
    parser.add_argument(
        "--accuracy-test",
        action="store_true",
        help="Enable accuracy testing mode with ground truth tracking and post-run analysis"
    )
    return parser.parse_args()


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

    # Match the number of features the scaler was fit on. Legacy models (NNetwork,
    # ScentCNN) were trained on 40 padded features; train_v2 MLPs use the real
    # feature count with no padding.
    expected = getattr(scaler, 'n_features_in_', 40)
    while features.shape[1] < expected:
        features[f'pad_{features.shape[1]}'] = 0.0

    X = scaler.transform(features.iloc[:, :expected])
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
    args = parse_args()

    # Load model + preprocessing
    model_path = input(
        "Enter model path or press Enter for default: "
    ).strip() or os.path.join(current_dir, "../02 training/model/scent_classifier.pth")
    
    print(f"Loading Model from path: {model_path}")
    
    model, scaler, label_encoder = load_model(path=model_path, device=DEVICE)

    # Initial Ground Truth Prompt if accuracy-test mode
    current_ground_truth = None
    if args.accuracy_test:
        print("\n" + "=" * 55)
        print("ACCURACY TESTING MODE ACTIVE")
        print("For each label, 5 rounds (50 samples) are collected and classified,")
        print("then you are prompted for the next label. Enter 'q' to stop & analyze.")
        print("=" * 55)
        while not current_ground_truth:
            current_ground_truth = input("Enter initial ground truth label: ").strip()
            if not current_ground_truth:
                print("Ground truth label cannot be empty.")
        print(f"Initial ground truth set to: '{current_ground_truth}'\n")

    # Init collector
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    collector = SensorDataCollector(
        csv_file=os.path.join(current_dir, f'csv/{"accuracy_test" if args.accuracy_test else "infer_log"}_{timestamp}.csv'),
        sample_freq_hz=4,
        samples_per_label=10,
        warmup_time=1,
        log_mode='inference',
    )

    try:
        collector.warm_up()
        print("Entering real-time inference loop. CTRL+C to exit.\n")

        if args.accuracy_test:
            ROUNDS_PER_LABEL = 5
            SAMPLES_PER_ROUND = 10
            while True:
                # Run a fixed batch of classification rounds for the current label.
                print(f"\nCollecting {ROUNDS_PER_LABEL} rounds "
                      f"({ROUNDS_PER_LABEL * SAMPLES_PER_ROUND} samples) "
                      f"for ground truth '{current_ground_truth}'...\n")
                for round_idx in range(ROUNDS_PER_LABEL):
                    df_window = collect_window_as_df(
                        collector, label=current_ground_truth, num_samples=SAMPLES_PER_ROUND
                    )
                    best_class, best_conf, class_conf_dict = predict_window(
                        model, scaler, label_encoder, df_window
                    )
                    collector.log_inference_result(
                        df_window,
                        best_class,
                        best_conf,
                        class_conf_dict,
                        label=current_ground_truth,
                    )
                    pretty_print_confidences(class_conf_dict, best_class)
                    is_match = (best_class.lower() == current_ground_truth.lower())
                    status_symbol = "MATCH" if is_match else "MISMATCH"
                    print(f"[Round {round_idx + 1}/{ROUNDS_PER_LABEL} | "
                          f"Ground Truth: '{current_ground_truth}' | "
                          f"Prediction: '{best_class}' -> {status_symbol}]\n")

                # Prompt for the next label before starting the next batch.
                try:
                    new_label = input("Enter next ground truth label (or 'q' to stop & analyze): ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting accuracy test...")
                    break
                while not new_label:
                    try:
                        new_label = input("Label cannot be empty. Enter ground truth (or 'q' to stop): ").strip()
                    except (KeyboardInterrupt, EOFError):
                        new_label = 'q'
                        break
                if new_label.lower() == 'q':
                    print("Exiting accuracy test...")
                    break
                current_ground_truth = new_label
                print(f"Ground truth updated to: '{current_ground_truth}'.")
        else:
            while True:
                df_window = collect_window_as_df(collector, label='', num_samples=10)
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
        if args.accuracy_test:
            try:
                from analysis import analyze_accuracy_log
                result = analyze_accuracy_log(collector.csv_file)
                
                if result:
                    with open(collector.csv_file, 'a', encoding='utf-8') as f:
                        f.write(result)
                    print(f"\Analysis results saved to: {collector.csv_file}")
            except ImportError:
                print("Error: Could not import analyze_accuracy_log from analysis module.")
            except Exception as e:
                print(f"Error during post-run analysis: {e}")


if __name__ == "__main__":
    main()

