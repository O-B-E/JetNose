"""
Sensor Data Collection and CSV Logging Utility

Provides the SensorDataCollector helper class to coordinate I²C sensor access
(temperature, humidity, and multi-gas readings), handle CSV file creation and logging,
and run interactive warm-up and purge baseline checks during sampling and inference.
"""

import sys
import os
import time
import csv
import json
import queue
import threading
from datetime import datetime
from smbus2 import SMBus

try:
    from aht import AHT2x
    from mgsv2 import MGSV2
except ImportError as e:
    print(f"Error: Could not import sensor drivers. {e}")
    sys.exit(1)


class SensorDataCollector:
    """
    Helper class for managing sensor hardware communication, sample collection, and logging.

    Coordinates reading from AHT2x temperature/humidity sensors and MGSV2 multi-gas
    sensors, writing measurements and model predictions to CSV files.
    """

    def __init__(self, csv_file=None, sample_freq_hz=4, samples_per_label=100, warmup_time=1):
        """
        Initialize SensorDataCollector and set up sensors and CSV log file.

        Parameters:
            csv_file (str, optional): Target CSV filepath. If None, generates timestamped filename in ../csv/
            sample_freq_hz (float): Data sampling frequency in Hertz (default 4 Hz)
            samples_per_label (int): Default number of samples to collect per label (default 100)
            warmup_time (int): Sensor preheat and warm-up duration in seconds (default 1)
        """
        self.sample_interval = 1.0 / sample_freq_hz
        self.samples_per_label = samples_per_label
        self.warmup_time = warmup_time
        
        # Headers for CSV
        self.headers = ['Timestamp', 'Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V', 'Label', 'Prediction', 'Confidence', 'ClassProbabilities']
        
        # Initialize CSV file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_file = csv_file or os.path.join(os.path.dirname(__file__), f'../csv/training_data_{timestamp}.csv')

        # Asynchronous inference logging queue and worker thread
        self.inference_queue = queue.Queue()
        self._async_log_running = True
        self.log_thread = threading.Thread(target=self._async_log_worker, daemon=True)
        self.log_thread.start()

        # Initialize sensors
        self.bus = None
        self.temp_sensor = None
        self.gas_sensor = None
        self._initialize_sensors()
        
        # Ensure CSV exists with headers
        self._ensure_csv_headers()
    
    def _initialize_sensors(self):
        """
        Internal method to initialize I²C bus and connect temperature and gas sensors.
        """
        try:
            self.bus = SMBus(1)
            self.temp_sensor = AHT2x(self.bus)
            self.gas_sensor = MGSV2(self.bus)
            print("Sensors initialized.")
        except Exception as e:
            print(f"Error initializing sensors: {e}")
            sys.exit(1)
    
    def _ensure_csv_headers(self):
        """
        Internal method to create target CSV file with header row if it does not exist.
        """
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.headers).writeheader()

    def _async_log_worker(self):
        """
        Internal background worker thread for writing inference log entries to CSV asynchronously.
        """
        while self._async_log_running or not self.inference_queue.empty():
            try:
                row = self.inference_queue.get(timeout=0.5)
                self._ensure_csv_headers()
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.headers)
                    writer.writerow(row)
                    f.flush()
                self.inference_queue.task_done()
            except queue.Empty:
                continue

    def warm_up(self):
        """
        Run sensor warm-up sequence with a console progress bar.

        Displays real-time gas sensor voltage readings until the specified warmup_time passes
        or until interrupted by CTRL+C.
        """
        print(f"\n{'='*80}")
        print(f"SENSOR WARM-UP ({self.warmup_time}s)")
        print("Press CTRL+C to skip")
        print(f"{'='*80}")
        
        start_time = time.time()
        try:
            while (time.time() - start_time) < self.warmup_time:
                elapsed = int(time.time() - start_time)
                remaining = self.warmup_time - elapsed
                
                try:
                    r = self.gas_sensor.read_all()
                    status = f"NO2:{r['NO2']:.2f}V Eth:{r['Ethanol']:.2f}V VOC:{r['VOC']:.2f}V CO:{r['CO']:.2f}V"
                except:
                    status = "Reading..."
                
                bar_len = 20
                filled = int(bar_len * elapsed / self.warmup_time)
                bar = '█' * filled + '-' * (bar_len - filled)
                print(f"\rWarmup: [{bar}] {remaining}s | {status}", end='')
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nWarm-up skipped.")
        
        print("\nWarm-up complete.\n")
    
    def purge_check(self):
        """
        Perform continuous sensor reading display to monitor baseline air recovery.

        Blocks and displays gas sensor voltages until user confirms baseline stability via CTRL+C.
        """
        print(f"\n{'!'*80}")
        print("PURGE CHECK: Wait for stable baseline")
        print("Press CTRL+C when ready")
        print(f"{'!'*80}")
        
        try:
            while True:
                r = self.gas_sensor.read_all()
                print(f"\rNO2:{r['NO2']:.2f}V Eth:{r['Ethanol']:.2f}V VOC:{r['VOC']:.2f}V CO:{r['CO']:.2f}V", end='')
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nBaseline confirmed.")
    
    def collect_sample(self, label):
        """
        Collect a single synchronous reading from temperature, humidity, and gas sensors.

        Parameters:
            label (str): Class or experiment label associated with the measurement

        Returns:
            dict: Dictionary containing timestamp, sensor values, and label
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        t, h = 0.0, 0.0
        if self.temp_sensor and self.temp_sensor.is_ready:
            t = self.temp_sensor.temperature
            h = self.temp_sensor.humidity
        
        gas = self.gas_sensor.read_all()
        
        return {
            'Timestamp': timestamp,
            'Temp': round(t, 2),
            'Hum': round(h, 2),
            'NO2_V': round(gas['NO2'], 2),
            'Eth_V': round(gas['Ethanol'], 2),
            'VOC_V': round(gas['VOC'], 2),
            'CO_V': round(gas['CO'], 2),
            'Label': label
        }
    
    def collect_samples(self, label, num_samples=None):
        """
        Collect multiple sensor readings at fixed intervals and save them to CSV file.

        Parameters:
            label (str): Class or experiment label associated with the dataset
            num_samples (int, optional): Number of samples to record (defaults to self.samples_per_label)

        Returns:
            list: List of dictionaries containing recorded sample data
        """
        num_samples = num_samples or self.samples_per_label
        
        print(f"Recording {num_samples} samples of '{label}'...")
        
        samples = []
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            
            for i in range(num_samples):
                cycle_start = time.time()
                
                data = self.collect_sample(label)
                writer.writerow(data)
                f.flush()
                samples.append(data)
                
                print(f"\rRec: {i+1}/{num_samples} | NO2:{data['NO2_V']:.2f}V Eth:{data['Eth_V']:.2f}V VOC:{data['VOC_V']:.2f}V CO:{data['CO_V']:.2f}V", end='')
                
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.sample_interval - elapsed)
                time.sleep(sleep_time)
        
        print(f"\nSaved {len(samples)} samples of '{label}' to {self.csv_file}")
        return samples
    
    def log_inference_result(self, df_window, prediction, confidence, class_conf_dict, label=''):
        """
        Append one inference window summary asynchronously to the CSV log queue.

        Calculates average sensor values over the window and queues prediction, confidence,
        and full class probability distributions for background writing without blocking.

        Parameters:
            df_window (pandas.DataFrame): Sliding window dataframe containing sensor readings
            prediction (str): Model's predicted class label
            confidence (float): Confidence score or probability of predicted class
            class_conf_dict (dict): Dictionary mapping class names to confidence probabilities
        """
        if df_window is None or df_window.empty:
            temp = hum = no2 = eth = voc = co = 0.0
        else:
            temp = round(float(df_window['Temp'].mean()), 2)
            hum = round(float(df_window['Hum'].mean()), 2)
            no2 = round(float(df_window['NO2_V'].mean()), 2)
            eth = round(float(df_window['Eth_V'].mean()), 2)
            voc = round(float(df_window['VOC_V'].mean()), 2)
            co = round(float(df_window['CO_V'].mean()), 2)

        row = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'Temp': temp,
            'Hum': hum,
            'NO2_V': no2,
            'Eth_V': eth,
            'VOC_V': voc,
            'CO_V': co,
            'Label': label,
            'Prediction': prediction,
            'Confidence': round(float(confidence), 2),
            'ClassProbabilities': "{" + ", ".join(f"{k}:{v:.2f}%" for k, v in class_conf_dict.items()) + "}"
        }

        # Non-blocking push to inference log queue
        self.inference_queue.put_nowait(row)

    def close(self):
        """
        Disconnect sensors, flush inference log queue, and release I²C bus resources.
        """
        # Signal background logging worker to finish and join thread
        if hasattr(self, '_async_log_running') and self._async_log_running:
            self._async_log_running = False
            if hasattr(self, 'log_thread') and self.log_thread.is_alive():
                self.log_thread.join(timeout=2.0)

        if self.bus:
            self.bus.close()
            print("Sensors disconnected.")

