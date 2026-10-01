"""
Multichannel Gas Sensor V2 Library

This module provides a class interface for Grove - Multichannel Gas Sensor V2
via I²C communication using smbus2.

Based on the official library https://github.com/Seeed-Studio/Seeed_Arduino_MultiGas
"""

import time

# GMXXX Gas Sensor I2C Constants
GM_ADDR = 0x08
GM_102B = 0x01  # NO2
GM_302B = 0x03  # Ethanol
GM_502B = 0x05  # VOC
GM_702B = 0x07  # CO
WARMING_UP = 0xFE

class MGSV2:
    def __init__(self, bus, address=GM_ADDR):
        self.bus = bus
        self.address = address
        self.preheat()

    def preheat(self):
        """Warm up the sensor array."""
        try:
            self.bus.write_byte(self.address, WARMING_UP)
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Preheat failed: {e}")

    def _read_sensor(self, cmd):
        """
        Reads 4 bytes from the sensor using Block Read logic.
        This is much more stable than reading byte-by-byte.
        """
        try:
            # Command to read specific sensor + read 4 bytes response
            # Format: 32-bit integer (Little Endian)
            data = self.bus.read_i2c_block_data(self.address, cmd, 4)
            
            if len(data) != 4:
                return 0.0

            # Convert bytes to integer
            val = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)
            
            # Convert ADC value to Voltage (0 - 3.3V)
            # Max ADC value is 1023 according to spec
            voltage = (val * 3.3) / 1023.0
            return voltage

        except Exception:
            return 0.0

    def read_no2(self):
        return self._read_sensor(GM_102B)

    def read_ethanol(self):
        return self._read_sensor(GM_302B)

    def read_voc(self):
        return self._read_sensor(GM_502B)

    def read_co(self):
        return self._read_sensor(GM_702B)

    def read_all(self):
        """Returns dictionary of all sensors in Volts."""
        return {
            'NO2': self.read_no2(),
            'Ethanol': self.read_ethanol(),
            'VOC': self.read_voc(),
            'CO': self.read_co()
        }