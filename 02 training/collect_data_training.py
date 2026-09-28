'''
collect_data_training.py

Interactive data-collection helper that uses `SensorDataCollector` to
warm sensors, prompt for labels, and record labeled samples into a CSV
for training.
'''

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../01 utils')
sys.path.append(utils_path)

from sensor_collector import SensorDataCollector

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive sensor data-collection."
    )
    parser.add_argument(
        '-s', '--samples',
        type=int,
        default=100,
        help="Number of samples to collect per label (default: 100)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    csv_filename = input("Enter CSV file name (default: training_data): ") or 'training_data'
    csv_file = os.path.join(current_dir, 'csv', f'{csv_filename}.csv')

    # Initialize collector (customize paths/settings here)
    collector = SensorDataCollector(
        csv_file=csv_file,
        sample_freq_hz=4,
        samples_per_label=args.samples,
        warmup_time=1
    )
    
    try:
        collector.warm_up()
        
        while True:
            collector.purge_check()
            
            print("-" * 80)
            label = input("Enter LABEL (or 'q' to quit): ").strip()
            if label.lower() == 'q':
                break
            if not label:
                continue
            
            print(f"Place '{label}' near sensors.")
            input("Press ENTER to record...")
            
            collector.collect_samples(label)
            print(f"Remove '{label}' sample now!\n")
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        collector.close()

if __name__ == "__main__":
    main()
