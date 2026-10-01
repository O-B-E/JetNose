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

# Constants for accuracy testing mode
ROUNDS_PER_LABEL = 5
SAMPLES_PER_ROUND = 15

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
    parser.add_argument(
        "-f", "--file",
        metavar="FILE_NAME",
        default=None,
        help="Run the accuracy test on pre-recorded samples in csv/<FILE_NAME>.csv "
             "(no live sensors needed) instead of collecting live data."
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
    Perform inference on a DataFrame window of sensor samples and return class probabilities.

    Dispatches on the model type:

    - Single-sample models (MLPs, ScentCNNsmall): each row in the window is
      classified independently and the softmax probabilities are averaged over the
      window (window = a smoothing device over independent per-row predictions).
    - Time-series model (ScentCNNrobust): the whole window is fed as a single
      sample shaped (1, sensors, window_length), producing one set of class
      probabilities directly (window = the atomic unit the model classifies).

    Parameters:
        model (nn.Module): Trained PyTorch model instance
        scaler (StandardScaler): Feature scaler instance used during model training
        label_encoder (LabelEncoder): Label encoder instance mapping indices to class names
        df_window (pandas.DataFrame): DataFrame containing window sensor readings

    Returns:
        tuple: (best_class, best_conf, class_conf_dict) where:
            best_class (str): Name of predicted class with highest confidence
            best_conf (float): Percentage confidence score for top predicted class
            class_conf_dict (dict): Dictionary mapping each class name to its confidence percentage
    """
    feature_cols = ['Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V']
    features = df_window[feature_cols].copy()
    X = scaler.transform(features)  # (num_rows, num_features)

    model.eval()
    with torch.no_grad():
        if getattr(model, 'time_series', False):
            # One window = one sample: (rows, sensors) -> (1, sensors, time).
            inputs = torch.FloatTensor(X).t().unsqueeze(0).to(DEVICE)
            logits = model(inputs)                      # (1, num_classes)
            conf_vals = F.softmax(logits, dim=1)[0].cpu().numpy()
        else:
            inputs = torch.FloatTensor(X).to(DEVICE)    # (rows, features)
            logits = model(inputs)                      # (rows, num_classes)
            probs = F.softmax(logits, dim=1)
            conf_vals = probs.mean(dim=0).cpu().numpy()  # average over the window

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


def pretty_print_confidences(class_conf_dict, best_class, num_samples=SAMPLES_PER_ROUND, aggregated=True):
    """
    Format and print per-class confidence percentages to the console.

    Parameters:
        class_conf_dict (dict): Dictionary mapping class names to confidence percentages
        best_class (str): Name of top predicted class to highlight with an arrow
        num_samples (int): Number of samples in the window that produced this prediction
        aggregated (bool): True when the scores are averaged over per-row predictions
            (single-sample models); False when the whole window is one sample
            (time-series model)
    """
    how = "samples aggregated" if aggregated else "sample window"
    print(f"\nWindow prediction ({num_samples} {how}):")
    for cls, conf in class_conf_dict.items():
        mark = "<--" if cls == best_class else "   "
        print(f"{cls:15s}: {conf:6.2f}% {mark}")
    print()


def append_model_footer(log_path, model_name, arch_name):
    """
    Append the model name and architecture footer to a completed log file.

    Parameters:
        log_path (str): Path to the log CSV to append to
        model_name (str): Model name used for this run
        arch_name (str): Architecture name of the model
    """
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\nModel: {model_name}\nArchitecture: {arch_name}\n")
    except Exception as e:
        print(f"Error writing model name to log: {e}")


def write_analysis_report(log_path, model_name, arch_name, timestamp):
    """
    Analyze an accuracy log and save a human-readable report alongside it.

    Parameters:
        log_path (str): Path to the accuracy log CSV to analyze
        model_name (str): Model name used for naming the report file
        arch_name (str): Architecture name recorded in the report header
        timestamp (str): Timestamp string used for naming the report file
    """
    try:
        from analysis import analyze_accuracy_log
        result = analyze_accuracy_log(log_path)
        if result:
            reports_dir = os.path.join(current_dir, 'analysis_reports')
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.join(reports_dir, f"report_{model_name}_{timestamp}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Model: {model_name}\nArchitecture: {arch_name}\n")
                f.write(result)
            print(f"\nAnalysis report saved to: {report_path}")
    except ImportError:
        print("Error: Could not import analyze_accuracy_log from analysis module.")
    except Exception as e:
        print(f"Error during post-run analysis: {e}")


def run_file_accuracy_test(model, arch_name, scaler, label_encoder, model_name, data_path):
    """
    Run the accuracy test on pre-recorded samples read from a CSV file.

    Rows are grouped into consecutive same-label segments (the file's 'Label' column is
    the ground truth), each segment is chunked into windows of SAMPLES_PER_ROUND samples,
    every window is classified, and the per-window results are written to a standard
    accuracy log which is then analyzed. No sensor hardware is required.

    Parameters:
        model (nn.Module): Trained PyTorch model instance
        arch_name (str): Architecture name (recorded in the log footer)
        scaler (StandardScaler): Feature scaler used during training
        label_encoder (LabelEncoder): Label encoder mapping indices to class names
        model_name (str): Model name used for naming the log/report files
        data_path (str): Path to the CSV file holding pre-recorded samples
    """
    df = pd.read_csv(data_path)
    if df.empty:
        print(f"Error: Sample file is empty: {data_path}")
        return
    if 'Label' not in df.columns:
        print(f"Error: Sample file '{data_path}' has no 'Label' column. Found: {list(df.columns)}")
        return

    df['Label'] = df['Label'].astype(str).str.strip()

    # Group into consecutive same-label segments so recovery-delay analysis stays meaningful.
    segment_id = (df['Label'] != df['Label'].shift()).cumsum()

    log_headers = ['Timestamp', 'Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V',
                   'Label', 'Prediction', 'Confidence', 'ClassProbabilities']
    log_rows = []
    total = correct = 0

    # The time-series model classifies one full window at a time, so chunk at its
    # trained window length and drop trailing partial windows (which are also too
    # short to survive the network's two max-pool stages). Single-sample models
    # keep the original fixed round size and accept any non-empty chunk.
    time_series = getattr(model, 'time_series', False)
    window_size = int(getattr(model, 'window_length', SAMPLES_PER_ROUND)) if time_series else SAMPLES_PER_ROUND

    print(f"\nRunning file-based accuracy test on: {os.path.normpath(data_path)}")
    print(f"Windowing {len(df)} samples into rounds of {window_size}"
          f"{' (full windows only)' if time_series else ''}...\n")

    for _, seg_df in df.groupby(segment_id, sort=False):
        label = str(seg_df['Label'].iloc[0])
        for start in range(0, len(seg_df), window_size):
            window = seg_df.iloc[start:start + window_size]
            if window.empty:
                continue
            if time_series and len(window) < window_size:
                # Skip a trailing partial window: the CNN expects full-length windows.
                continue

            best_class, best_conf, class_conf_dict = predict_window(
                model, scaler, label_encoder, window
            )

            # Preserve the window's real first timestamp so recovery delays reflect
            # the original recording timing; fall back to now() if absent/unparseable.
            ts = None
            if 'Timestamp' in window.columns:
                ts = str(window['Timestamp'].iloc[0])
            if not ts or ts == 'nan':
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            log_rows.append({
                'Timestamp': ts,
                'Temp': round(float(window['Temp'].mean()), 2),
                'Hum': round(float(window['Hum'].mean()), 2),
                'NO2_V': round(float(window['NO2_V'].mean()), 2),
                'Eth_V': round(float(window['Eth_V'].mean()), 2),
                'VOC_V': round(float(window['VOC_V'].mean()), 2),
                'CO_V': round(float(window['CO_V'].mean()), 2),
                'Label': label,
                'Prediction': best_class,
                'Confidence': round(float(best_conf), 2),
                'ClassProbabilities': "{" + ", ".join(
                    f"{k}:{v:.2f}%" for k, v in class_conf_dict.items()) + "}",
            })

            total += 1
            is_match = best_class.lower() == label.lower()
            correct += int(is_match)
            print(f"[{total:4d}] Ground Truth: '{label}' | "
                  f"Prediction: '{best_class}' ({best_conf:.1f}%) -> "
                  f"{'MATCH' if is_match else 'MISMATCH'}")

    if not log_rows:
        print("No windows were produced from the sample file.")
        return

    print(f"\nClassified {total} windows | running accuracy: "
          f"{100.0 * correct / total:.2f}% ({correct}/{total})")

    # Persist the per-window results as a standard accuracy log.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = os.path.join(current_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'accuracy_test_{model_name}_{timestamp}.csv')
    pd.DataFrame(log_rows, columns=log_headers).to_csv(log_path, index=False)
    append_model_footer(log_path, model_name, arch_name)
    print(f"Accuracy log saved to: {log_path}")

    # Reuse the shared analysis to produce the report.
    write_analysis_report(log_path, model_name, arch_name, timestamp)


def main():
    """
    Main execution loop for real-time sensor sampling, model loading, and live inference logging.
    """
    args = parse_args()

    # Load model + preprocessing
    while True:
        model_name = input("Enter model name (default: ScentCNNsmall): ") or 'ScentCNNsmall'
        model_path = os.path.join(current_dir, '../02 training/model', f'{model_name}.pth')
        if os.path.exists(model_path):
            break
        print(f'Model file not found: {model_path}. Please try again.')

    print(f"Loading Model from path: {os.path.normpath(model_path)}")

    model, arch_name, scaler, label_encoder = load_model(path=model_path, device=DEVICE)

    # For the time-series model, live windows must match its trained window length;
    # single-sample models use the fixed per-round sample count.
    is_time_series = getattr(model, 'time_series', False)
    round_size = int(getattr(model, 'window_length', SAMPLES_PER_ROUND)) if is_time_series else SAMPLES_PER_ROUND

    # File-based accuracy test: classify pre-recorded samples, no live sensors.
    while args.file:
        data_path = os.path.join(current_dir, 'csv', f'{args.file}.csv')
        if not os.path.exists(data_path):
            print(f"Error: Sample file not found: {os.path.normpath(data_path)}")
            data_path = input("Enter CSV file name (<file_name>.csv): ").strip()
            continue

        run_file_accuracy_test(model, arch_name, scaler, label_encoder, model_name, data_path)
        return

    # Initial Ground Truth Prompt if accuracy-test mode
    current_ground_truth = None
    if args.accuracy_test:
        print("\n" + "=" * 55)
        print("ACCURACY TESTING MODE ACTIVE")
        print(f"For each label, {ROUNDS_PER_LABEL} rounds of {SAMPLES_PER_ROUND} samples, {ROUNDS_PER_LABEL * SAMPLES_PER_ROUND} samples are collected and classified,")
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
        csv_file=os.path.join(current_dir, f'logs/{"accuracy_test" if args.accuracy_test else "infer_log"}_{model_name}_{timestamp}.csv'),
        sample_freq_hz=4,
        samples_per_label=10,
        warmup_time=1,
        log_mode='inference',
    )

    try:
        collector.warm_up()
        print("Entering real-time inference loop. CTRL+C to exit.\n")

        if args.accuracy_test:
            while True:
                # Run a fixed batch of classification rounds for the current label.
                print(f"\nCollecting {ROUNDS_PER_LABEL} rounds "
                      f"({ROUNDS_PER_LABEL * round_size} samples) "
                      f"for ground truth '{current_ground_truth}'...\n")
                for round_idx in range(ROUNDS_PER_LABEL):
                    df_window = collect_window_as_df(
                        collector, label=current_ground_truth, num_samples=round_size
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
                    pretty_print_confidences(class_conf_dict, best_class,
                                             num_samples=round_size, aggregated=not is_time_series)
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
                df_window = collect_window_as_df(collector, label='', num_samples=round_size)
                best_class, best_conf, class_conf_dict = predict_window(
                    model, scaler, label_encoder, df_window
                )
                collector.log_inference_result(
                    df_window,
                    best_class,
                    best_conf,
                    class_conf_dict,
                )
                pretty_print_confidences(class_conf_dict, best_class,
                                         num_samples=round_size, aggregated=not is_time_series)

    except KeyboardInterrupt:
        print("\nInference stopped by user.")
    finally:
        collector.close()

        # Record which model produced this log, after all samples are written
        # (collector.close() flushes them) and before any analysis output.
        append_model_footer(collector.csv_file, model_name, arch_name)

        if args.accuracy_test:
            write_analysis_report(collector.csv_file, model_name, arch_name, timestamp)


if __name__ == "__main__":
    main()

