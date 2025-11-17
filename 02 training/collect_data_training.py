import sys
import os
import time
import csv
from datetime import datetime
from smbus2 import SMBus

# Importing utils
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '../utils')
sys.path.append(utils_path)

try:
    from aht import AHT2x
    from mgsv2 import MGSV2
except ImportError as e:
    print(f"\nError: Could not import drivers from {utils_path}")
    print(f"Details: {e}")
    sys.exit(1)


# Sampling settings
SAMPLE_FREQ_HZ = 4
SAMPLE_INTERVAL = 1.0 / SAMPLE_FREQ_HZ 
SAMPLES_PER_LABEL = 100
WARMUP_TIME_SECONDS = 1 #900
CSV_FILE = os.path.join(current_dir, 'csv/training_data.csv')

# Global variables
bus = None
temp_sensor = None
gas_sensor = None


def initialize_sensors():
    global bus, temp_sensor, gas_sensor
    print("Initializing sensors...")
    try:
        bus = SMBus(1)
        temp_sensor = AHT2x(bus)
        gas_sensor = MGSV2(bus)
        print("Sensors connected.")
    except Exception as e:
        print(f"Error initializing I2C: {e}")
        sys.exit(1)

def warm_up_sequence():
    
    print(f"\n{'='*80}")
    print(f"SENSOR WARM-UP SEQUENCE ({WARMUP_TIME_SECONDS} seconds)")
    print("Press CTRL+C to SKIP warm-up (only if sensors are already hot!)")
    print(f"{'='*80}")

    start_time = time.time()
    try:
        while (time.time() - start_time) < WARMUP_TIME_SECONDS:
            elapsed = int(time.time() - start_time)
            remaining = WARMUP_TIME_SECONDS - elapsed
            
            try:
                r = gas_sensor.read_all()
                status = f"NO2: {r['NO2']:.2f}V | Eth: {r['Ethanol']:.2f}V | VOC: {r['VOC']:.2f}V | CO: {r['CO']:.2f}V"
            except: 
                status = "Reading..."

            bar_len = 20
            filled = int(bar_len * elapsed / WARMUP_TIME_SECONDS)
            bar = '█' * filled + '-' * (bar_len - filled)
            
            print(f"\rWarmup: [{bar}] {remaining}s | {status}   ", end='')
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n>>> Warm-up skipped by user.")
        time.sleep(1)
        return

    print("\n\nWarm-up complete.\n")

def purge_sensors_display():
    """ Display the sensor data between 2 sample sessions """
    print(f"\n{'!'*80}")
    print("PURGE CHECK: Wait for ALL values to drop and stabilize.")
    print("Press CTRL+C when values are stable to start recording.")
    print(f"{'!'*80}")
    
    try:
        while True:
            r = gas_sensor.read_all()
            print(f"\rNO2: {r['NO2']:.2f}V | Eth: {r['Ethanol']:.2f}V | VOC: {r['VOC']:.2f}V | CO: {r['CO']:.2f}V   ", end='')
            time.sleep(0.25)
            
    except KeyboardInterrupt:
        print("\n\nBaseline confirmed. Starting setup...")
        time.sleep(0.5)
        try:
            sys.stdin.flush()
        except: pass

def collect_sample(label):
    ''' Sample the sensor data and return it to the csv file '''
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    t, h = 0.0, 0.0
    if temp_sensor.is_ready:
        t = temp_sensor.temperature
        h = temp_sensor.humidity
    
    gas = gas_sensor.read_all()
    
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

def main():
    initialize_sensors()
    warm_up_sequence()
    
    headers = ['Timestamp', 'Temp', 'Hum', 'NO2_V', 'Eth_V', 'VOC_V', 'CO_V', 'Label']
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    while True:
        purge_sensors_display()
        
        print("-" * 80)
        label = input("Enter LABEL for next sample (or 'q' to quit): ").strip()
        if label.lower() == 'q': break
        if not label: continue

        print(f"Place '{label}' near sensors.")
        input("Press ENTER to start recording (25 seconds)...")
        
        print(f"Recording {SAMPLES_PER_LABEL} samples...")
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            
            for i in range(SAMPLES_PER_LABEL):
                cycle_start = time.time()
                
                data = collect_sample(label)
                writer.writerow(data)
                f.flush()
                
                print(f"\rRec: {i+1}/{SAMPLES_PER_LABEL} | NO2: {data['NO2_V']:.2f}V | Eth: {data['Eth_V']:.2f}V | VOC: {data['VOC_V']:.2f}V | CO: {data['CO_V']:.2f}V", end='')
                
                elapsed = time.time() - cycle_start
                sleep_time = max(0, SAMPLE_INTERVAL - elapsed)
                time.sleep(sleep_time)

        print(f"\nSaved '{label}'. Remove sample now!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped.")
        if bus: bus.close()