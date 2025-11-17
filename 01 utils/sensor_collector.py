import sys
import os
import time
import csv
from datetime import datetime
from smbus2 import SMBus

try:
    from aht import AHT2x
    from mgsv2 import MGSV2
except ImportError as e:
    print(f"Error: Could not import sensor drivers. {e}")
    sys.exit(1)

class SensorDataCollector:
    def __init__(self, csv_file=None, sample_freq_hz=4, samples_per_label=100, warmup_time=1):
        self.sample_interval = 1.0 / sample_freq_hz
        self.samples_per_label = samples_per_label
        self.warmup_time = warmup_time
        
        # Headers for CSV
        self.headers = ['Timestamp', 'Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V', 'Label']
        
        # Initialize sensors
        self.csv_file = csv_file or os.path.join(os.path.dirname(__file__), '../csv/training_data.csv')
        self.bus = None
        self.temp_sensor = None
        self.gas_sensor = None
        self._initialize_sensors()
        
        # Ensure CSV exists with headers
        self._ensure_csv_headers()
    
    def _initialize_sensors(self):
        """Initialize I2C bus and sensors"""
        try:
            self.bus = SMBus(1)
            self.temp_sensor = AHT2x(self.bus)
            self.gas_sensor = MGSV2(self.bus)
            print("Sensors initialized.")
        except Exception as e:
            print(f"Error initializing sensors: {e}")
            sys.exit(1)
    
    def _ensure_csv_headers(self):
        """Create CSV file with headers if it doesn't exist"""
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.headers).writeheader()
    
    def warm_up(self):
        """Warm-up sequence with progress bar"""
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
        """Wait for stable baseline readings"""
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
        """Collect single sensor reading"""
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
        """Collect multiple samples and save to CSV"""
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
    
    def close(self):
        """Clean up resources"""
        if self.bus:
            self.bus.close()
            print("Sensors disconnected.")
