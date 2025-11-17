import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../01 utils')
sys.path.append(utils_path)

from sensor_collector import SensorDataCollector

def main():
    # Initialize collector (customize paths/settings here)
    collector = SensorDataCollector(
        csv_file='csv/training_data.csv',
        sample_freq_hz=4,
        samples_per_label=100,
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
