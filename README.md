# JetNose 👃

**An all-in-one electronic nose on an NVIDIA Jetson Nano.** It collects odor samples, trains a deep-learning classifier and identifies odors in real time, all on the same device.

Most e-nose systems collect data on one machine and train on another, which makes it hard to fine-tune them, add new odors or adapt to a new environment. JetNose keeps the whole loop on the Jetson: an array of four MOS gas sensors plus a temperature/humidity sensor gives six input channels, and four neural networks (two MLPs, two 1D CNNs) are trained and compared on them.

<p align="center">
  <img src="02%20training/figures/training_samples_visuals/pca_3d.png" width="520" alt="PCA of the training samples">
  <br><em>The four odors in the first three principal components of the training data.</em>
</p>

---

## Highlights

- **Fully on-device:** data collection → training → live inference on a Jetson Nano.
- **6 sensor channels:** NO₂, ethanol, VOC and CO (Grove Multichannel Gas Sensor V2) + temperature and humidity (AHT20/21), sampled at 4 Hz over I²C.
- **4 odors:** air (control), coffee, alcohol, lemon. 4,000 balanced training samples.
- **4 models** compared on a separately recorded test set (different location and conditions).
- **Statistical evaluation:** segment-bootstrap 95% CIs, macro-F1, McNemar tests, recovery delay and calibration (ECE).

## Results

| Model | Params | Accuracy [95% CI] | Macro-F1 [95% CI] | ECE |
|---|---:|---|---|---:|
| ScentMLPsmall   |    820 | 81.8% [65.0–95.0] | 81.3% [65.8–94.4] | 11.8% |
| ScentMLProbust  | 11,812 | 82.9% [65.0–96.4] | 81.8% [65.5–96.0] | 16.2% |
| ScentCNNsmall   |  1,860 | 70.4% [52.5–86.1] | 69.9% [52.6–84.2] | 20.8% |
| **ScentCNNrobust** | 90,148 | **83.8%** [69.2–95.4] | **83.8%** [68.1–94.5] | **9.7%** |

- **ScentCNNrobust**, which convolves along the *time* axis, is the most accurate and best calibrated model. Its lead over the MLPs is **not statistically significant**, though the small MLP has under 1% of its parameters.
- **ScentCNNsmall**, which convolves across the (arbitrarily ordered) sensor channels, is significantly worse than all the others (McNemar p < 0.001).
- About 90% of errors come from two confusions: **lemon → air** and **coffee → lemon**. The main challenge is robustness to changing conditions, not model capacity.

<p align="center">
  <img src="03%20inference/figures/ScentMLPsmall_ScentMLProbust_ScentCNNsmall_ScentCNNrobust_2026-10-10_16-12-59/1_accuracy_ci.png" width="48%" alt="Accuracy and macro-F1 with 95% CIs">
  <img src="03%20inference/figures/ScentMLPsmall_ScentMLProbust_ScentCNNsmall_ScentCNNrobust_2026-10-10_16-12-59/2_confusion.png" width="48%" alt="Confusion matrices">
</p>
<p align="center">
  <img src="02%20training/figures/ScentMLPsmall_ScentMLProbust_ScentCNNsmall_ScentCNNrobust_2026-10-10_10-37-42/training_loss.png" width="70%" alt="Training and validation loss">
</p>

Full reports are in [`03 inference/analysis_reports/`](03%20inference/analysis_reports/) and all figures in [`03 inference/figures/`](03%20inference/figures/).

## Hardware

| Component | Role |
|---|---|
| NVIDIA Jetson Nano | Collection, training and inference |
| [Grove Multichannel Gas Sensor V2](https://github.com/Seeed-Studio/Seeed_Arduino_MultiGas) (I²C) | 4 MOS sensors: NO₂ (GM-102B), ethanol (GM-302B), VOC (GM-502B), CO (GM-702B) |
| [AHT25](https://github.com/adafruit/Adafruit_AHTX0) (I²C) | Temperature and humidity |

## Models

| Model | Input | Architecture |
|---|---|---|
| `ScentMLPsmall`  | single reading (6 features) | 6 → 32 → 16 → 4, ReLU + Dropout 0.3 |
| `ScentMLProbust` | single reading (6 features) | 6 → 128 → 64 → 32 → 4, BatchNorm + ReLU + Dropout 0.4 |
| `ScentCNNsmall`  | single reading as a 1-channel sequence | 2 × Conv1d (16, 32) + BN → global avg pool → FC |
| `ScentCNNrobust` | window of *W* readings (default 15), sensors as channels | 4 × Conv1d (32, 64, 128, 128) + BN, max-pool, dropout → global avg pool → FC |

Training uses a **leakage-safe grouped split**: readings are grouped by recording run, so near-duplicate neighbors never land in both train and validation sets. The scaler is fit on the training split only, and the best-validation-loss checkpoint is kept. Each `.pth` checkpoint stores the weights together with the scaler and label encoder.

## Repository structure

```
JetNose/
├── 01 utils/
│   ├── aht.py               # AHT2x temperature/humidity driver 
│   ├── mgsv2.py             # Grove Multichannel Gas Sensor V2 driver 
│   ├── sensor_collector.py  # Sampling, warm-up/purge checks, CSV logging
│   ├── model_arch.py        # The four architectures, windowing, load_model()
│   ├── analysis.py          # Accuracy-log statistics and model comparison reports
│   └── plots.py             # All result, training and data figures
├── 02 training/
│   ├── collect_data_training.py  # Interactive labeled data collection
│   ├── train_MLP.py              # Train ScentMLPsmall / ScentMLProbust
│   ├── train_CNN.py              # Train ScentCNNsmall / ScentCNNrobust
│   ├── csv/       # Collected training data
│   ├── model/     # Trained checkpoints (.pth)
│   ├── logs/      # Per-epoch training histories
│   └── figures/   # Loss curves, per-odor sensor plots, PCA
└── 03 inference/
    ├── infer.py              # Live inference and accuracy testing
    ├── csv/                  # Recorded test data
    ├── logs/                 # Accuracy-test logs (one per model)
    ├── analysis_reports/     # Text reports from analysis.py
    └── figures/              # Accuracy, confusion, timeline, calibration plots
```

## Getting started

### Requirements

Python 3.6+ (the Jetson Nano JetPack default) with:

```bash
pip install torch numpy pandas scikit-learn matplotlib smbus2
```

On the Jetson, install the NVIDIA-provided PyTorch wheel for your JetPack version, and preload OpenMP before running inference:

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

`smbus2` and the sensors are only needed for live collection and inference. Training, file-based accuracy tests, analysis and plotting run on any machine.

### 1. Collect training data

```bash
cd "02 training"
python collect_data_training.py --samples 100
```

The script warms the sensors up, then repeatedly asks for a label (e.g. `coffee`), waits for you to place the sample, and records `--samples` readings at 4 Hz into `csv/<name>.csv`. A purge check before each label makes sure the sensors have returned to baseline.

### 2. Train a model

```bash
python train_MLP.py   # prompts: CSV name, small/robust, model name
python train_CNN.py   # robust also prompts for window length and stride
```

Checkpoints are saved to `02 training/model/<name>.pth` and the epoch history to `02 training/logs/`.

### 3. Run inference

```bash
cd "03 inference"
python infer.py                                         # live classification
python infer.py --accuracy-test                         # live, with ground-truth labels you enter
python infer.py --accuracy-test -f accuracy_test_data   # replay a recorded CSV, no sensors needed
```

You are prompted for the model name (e.g. `ScentCNNrobust`). Predictions and confidences are logged to `03 inference/logs/`.

### 4. Analyze and plot

```bash
cd "01 utils"
python analysis.py accuracy_test_ScentMLPsmall_2026-09-28_17-39-37.csv            # one model
python analysis.py <log_a.csv> <log_b.csv> ...                                     # compare models
python plots.py <log_a.csv> <log_b.csv> ...                                        # all accuracy figures
python plots.py -l <log_a.csv> <log_b.csv> ...                                     # training-loss figure
python plots.py --visualize training_data.csv                                      # per-odor sensor plots + PCA
```

Bare log names are looked up in `03 inference/logs/` (or `02 training/logs/` for training histories). Figures are written as 300-dpi PNGs to a new timestamped folder with a `sources.txt` listing their inputs. Use `-a -c -p -t -d -r` to make only the accuracy, confusion, per-class, timeline, delay or calibration figures.
