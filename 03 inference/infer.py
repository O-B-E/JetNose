'''
! IMPORTANT !
Before running this script export the following environment variable:
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
'''

import os
import sys
import time
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
    print('Details: {e}')
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collect_window_as_df(collector, label="inference", num_samples=10):
    """
    Collect num_samples from sensors and return as a pandas DataFrame
    shaped like the training CSV (but Label is dummy/unused).
    """
    rows = []
    for i in range(num_samples):

        data = collector.collect_sample(label)
        rows.append(data)

        print(
            f"\rSample {i+1}/{num_samples} | "
            f"NO2:{data['NO2_V']:.2f}V Eth:{data['Eth_V']:.2f}V "
            f"VOC:{data['VOC_V']:.2f}V CO:{data['CO_V']:.2f}V",
            end='',
        )

        time.sleep(0.25)

    print()
    return pd.DataFrame(rows)

def predict_window(model, scaler, label_encoder, df_window):
    """
    Run model on a DataFrame of N samples and aggregate over the window:
    - per-class mean probability over the N samples
    - final predicted class and confidence in percent
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
    print("\nWindow prediction (10 samples aggregated):")
    for cls, conf in class_conf_dict.items():
        mark = "<--" if cls == best_class else "   "
        print(f"{cls:15s}: {conf:6.2f}% {mark}")
    print()

def main():
    # Load model + preprocessing
    model_path = input(
        "Enter model path or press Enter for default: "
    ).strip() or os.path.join(current_dir, "../02 training/model/scent_classifier.pth")
    model, scaler, label_encoder = load_model(path=model_path, device=DEVICE)

    # Init collector (no CSV needed for inference, but reuse class)
    collector = SensorDataCollector(
        csv_file=os.path.join(current_dir, 'csv/infer_log.csv'),
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
            pretty_print_confidences(class_conf_dict, best_class)

    except KeyboardInterrupt:
        print("\nInference stopped by user.")
    finally:
        collector.close()

if __name__ == "__main__":
    main()
