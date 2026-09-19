"""
Analysis Module for Inference Accuracy and Recovery Delay Evaluation

Computes overall classification accuracy and transition recovery delay per label
from inference CSV log files generated during accuracy testing mode.
"""

import os
import sys
import pandas as pd
from datetime import datetime


def analyze_accuracy_log(log_file_path):
    """
    Analyze inference accuracy test log file.

    Computes:
      1. Overall accuracy = (Total Correct Predictions / Total Predictions) * 100%
      2. Recovery delay per label segment:
         - Segment starts when Ground Truth transitions to a label.
         - If initial prediction != Ground Truth, delay state starts at first prediction.
         - Delay state ends at first prediction where Predicted Label == Ground Truth.
         - Computes mean recovery delay (seconds) aggregated per label across occurrences.

    Parameters:
        log_file_path (str): Path to CSV log file.

    Returns:
        dict: Dictionary containing summary metrics and per-label statistics.
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found: {log_file_path}")
        return None

    try:
        df = pd.read_csv(log_file_path)
    except Exception as e:
        print(f"Error reading log file '{log_file_path}': {e}")
        return None

    if df.empty:
        print(f"Warning: Log file '{log_file_path}' is empty.")
        return None

    # Detect column names (case-insensitive with fallback)
    col_map = {c.strip().lower(): c for c in df.columns}

    gt_col = col_map.get('label')
    pred_col = col_map.get('prediction')
    time_col = col_map.get('timestamp')

    if not gt_col or not pred_col:
        print(f"Error: Missing required columns in '{log_file_path}'. Found: {list(df.columns)}")
        return None

    # Filter valid rows
    df = df.dropna(subset=[gt_col, pred_col]).copy()
    if df.empty:
        print(f"Warning: No valid prediction rows found in '{log_file_path}'.")
        return None

    df[gt_col] = df[gt_col].astype(str).str.strip()
    df[pred_col] = df[pred_col].astype(str).str.strip()

    # Parse timestamps
    if time_col and time_col in df.columns:
        df['parsed_time'] = pd.to_datetime(df[time_col], errors='coerce')
    else:
        df['parsed_time'] = pd.NaT

    # 1. Overall Accuracy
    total_predictions = len(df)
    is_correct = df[gt_col].str.lower() == df[pred_col].str.lower()
    total_correct = int(is_correct.sum())
    overall_accuracy = (total_correct / total_predictions * 100.0) if total_predictions > 0 else 0.0

    # 2. Segment Detection & Recovery Delay per Occurrence
    # Transition occurs whenever Ground Truth changes
    segment_id = (df[gt_col].str.lower() != df[gt_col].str.lower().shift(1)).cumsum()
    df['segment_id'] = segment_id

    segment_results = []
    delays_by_label = {}  # label -> list of delay seconds for occurrences with initial misclassification
    all_delays_by_label = {}  # label -> list of delay seconds including 0s for immediate matches

    for seg_id, seg_df in df.groupby('segment_id'):
        label = seg_df[gt_col].iloc[0]
        seg_predictions = seg_df[pred_col].tolist()
        seg_times = seg_df['parsed_time'].tolist()

        if label not in delays_by_label:
            delays_by_label[label] = []
            all_delays_by_label[label] = []

        first_pred = seg_predictions[0]
        first_correct = (first_pred.lower() == label.lower())

        if first_correct:
            # Matched immediately: 0 recovery delay
            seg_info = {
                'segment_id': seg_id,
                'label': label,
                'initial_match': True,
                'recovered': True,
                'delay_seconds': 0.0,
                'num_samples': len(seg_df)
            }
            all_delays_by_label[label].append(0.0)
            segment_results.append(seg_info)
        else:
            # Initial misclassification occurred: delay state starts at first prediction
            t_start = seg_times[0]
            recovered = False
            delay_sec = None

            # Find first index where prediction == ground truth
            for idx, (pred, t_curr) in enumerate(zip(seg_predictions, seg_times)):
                if pred.lower() == label.lower():
                    recovered = True
                    if pd.notna(t_start) and pd.notna(t_curr):
                        delay_sec = (t_curr - t_start).total_seconds()
                    else:
                        # Fallback if timestamps missing: window count approximation
                        delay_sec = float(idx)
                    break

            seg_info = {
                'segment_id': seg_id,
                'label': label,
                'initial_match': False,
                'recovered': recovered,
                'delay_seconds': delay_sec,
                'num_samples': len(seg_df)
            }

            if recovered and delay_sec is not None:
                delays_by_label[label].append(delay_sec)
                all_delays_by_label[label].append(delay_sec)

            segment_results.append(seg_info)

    # 3. Aggregate Mean Delay per Label
    label_summary = {}
    for label in all_delays_by_label:
        misclass_delays = delays_by_label.get(label, [])
        all_delays = all_delays_by_label.get(label, [])

        mean_misclass_delay = (sum(misclass_delays) / len(misclass_delays)) if misclass_delays else None
        mean_all_delay = (sum(all_delays) / len(all_delays)) if all_delays else 0.0

        label_summary[label] = {
            'occurrences': len(all_delays),
            'misclassified_occurrences': len(misclass_delays),
            'mean_recovery_delay_sec': mean_misclass_delay if mean_misclass_delay is not None else 0.0,
            'mean_overall_delay_sec': mean_all_delay
        }

    # 4. Formatted Terminal Display
    lines = [
        f"\n\n{'=' * 65}",
        "ACCURACY TEST ANALYSIS",
        f"{'=' * 65}\n",
        f"{'Segment':<9} | {'Ground Truth':<15} | {'Initial':<11} | {'Recovery Delay':<15}",
        "-" * 65,
    ]

    for s in segment_results:
        init_str = "MATCH" if s['initial_match'] else "MISMATCH"
        delay_str = f"{s['delay_seconds']:.2f}s" if s['delay_seconds'] is not None else "NO RECOVERY"
        lines.append(f"{s['segment_id']:<9} | {s['label']:<15} | {init_str:<11} | {delay_str:<15}")

    lines.extend([
        "-" * 65,
        f"{'Label':<15} | {'Occurrences':<12} | {'Initial Mismatches':<20} | {'Mean Recovery Delay':<20}",
        "-" * 65,
    ])

    for lbl, stats in label_summary.items():
        occ = stats['occurrences']
        mismatches = stats['misclassified_occurrences']
        mean_delay = stats['mean_recovery_delay_sec']
        lines.append(f"{lbl:<15} | {occ:<12} | {mismatches:<20} | {mean_delay:.2f}s")

    lines.append("=" * 65 + "\n")

    analysis_output = "\n".join(lines)
    print(analysis_output)

    return analysis_output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        analyze_accuracy_log(log_path)
    else:
        print("Usage: python analysis.py <path_to_log.csv>")
