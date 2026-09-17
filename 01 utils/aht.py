"""
AHT2x Temperature and Humidity Sensor Library

This module provides a class interface for AHT20 and AHT21 sensors
via I²C communication using smbus2.

Based on the official library https://github.com/adafruit/Adafruit_AHTX0
"""

import time
from smbus2 import SMBus

# AHT2x I²C Constants
AHT_I2C_ADDR = 0x38  # Default I2C address
AHT_STATUS_BUSY = 0x01  # Status bit for busy
AHT_STATUS_CALIBRATED = 0x10  # Status bit for calibrated
AHT_CMD_INIT = 0xBE  # Command for initialization
AHT_CMD_TRIGGER = 0xAC  # Command for trigger measurement
AHT_CMD_RESET = 0xBA  # Command for soft reset
AHT_CRC_POLYNOMIAL = 0x31  # Polynomial representation for CRC
AHT_CRC_MSB = 0x80  # Most significant bit for CRC
AHT_CRC_INIT = 0xFF  # Initial value of CRC


class AHT2x:
    """
    Class for AHT20 and AHT21 temperature and humidity sensors.
    
    Provides methods to read temperature and humidity with optional CRC validation.
    """

    def __init__(self, bus, address=AHT_I2C_ADDR, crc=False):
        """
        Initialize AHT2x sensor.
        
        Parameters:
            bus: Instance of smbus2.SMBus for I²C communication
            address: I²C address of sensor (default 0x38)
            crc: Boolean for CRC control. True to activate CRC validation
        """
        time.sleep(0.04)  # Wait 40ms after power-on for sensor stability
        self.bus = bus
        self.address = address
        self.active_crc = crc
        self._buf = bytearray(6 + crc)  # Buffer size includes optional CRC byte
        self.humidity = None
        self.temperature = None
        
        # Ensure sensor is calibrated before use
        while not self.is_calibrated:
            self._calibrate()

    @property
    def is_ready(self):
        """
        Check if sensor is ready and perform measurement.
        
        The sensor is busy until the measurement is complete.
        
        Returns:
            bool: True if measurement successful, False if sensor busy
        """
        if bool(self._status() & AHT_STATUS_BUSY):
            return False
        return self._measure()

    @property
    def is_calibrated(self):
        """
        Check if sensor is calibrated.
        
        Calibration must be done before any measurement. If not calibrated,
        perform a soft reset.
        
        Returns:
            bool: True if sensor is calibrated
        """
        return bool(self._status() & AHT_STATUS_CALIBRATED)

    def _status(self):
        """
        Read and return the status byte from sensor.
        
        Returns:
            int: Status byte, or AHT_STATUS_BUSY if CRC check fails
        """
        # Read 6 bytes (or 7 with CRC) from sensor
        self._buf = self.bus.read_i2c_block_data(self.address, 0x00, 6 + self.active_crc)

        # Validate CRC if enabled
        if not self.active_crc or (self._crc8() == self._buf[6]):
            return self._buf[0]
        return AHT_STATUS_BUSY  # Return busy status if CRC fails

    def reset(self):
        """
        Perform soft reset of sensor.
        
        The soft reset command restarts the sensor system without
        power cycling.
        """
        self._buf[0] = AHT_CMD_RESET
        self.bus.write_i2c_block_data(self.address, 0x00, self._buf[:1])
        time.sleep(0.02)  # Reset time does not exceed 20ms

        # Re-calibrate after reset
        while not self.is_calibrated:
            self._calibrate()

    def _calibrate(self):
        """
        Internal function to send initialization/calibration command.
        """
        self._buf[0] = AHT_CMD_INIT
        self._buf[1] = 0x08
        self._buf[2] = 0x00
        self.bus.write_i2c_block_data(self.address, 0x00, self._buf[:3])
        time.sleep(0.01)  # Wait for initialization process

    def _crc8(self):
        """
        Internal function to calculate CRC-8-Dallas/Maxim checksum.
        
        Returns:
            int: Calculated CRC-8 value
        """
        crc = bytearray(1)
        crc[0] = AHT_CRC_INIT
        
        # Calculate CRC for first 6 bytes
        for byte in self._buf[:6]:
            crc[0] ^= byte
            for _ in range(8):
                if crc[0] & AHT_CRC_MSB:
                    crc[0] = (crc[0] << 1) ^ AHT_CRC_POLYNOMIAL
                else:
                    crc[0] = crc[0] << 1

        return crc[0]

    def _measure(self):
        """
        Internal function for triggering measurement and reading data.
        
        Sends trigger command, waits for measurement completion, then
        reads and parses temperature and humidity values.
        
        Returns:
            bool: True if measurement successful, False if CRC fails
        """
        # Send trigger measurement command
        self._buf[0] = AHT_CMD_TRIGGER
        self._buf[1] = 0x33
        self._buf[2] = 0x00
        self.bus.write_i2c_block_data(self.address, 0x00, self._buf[:3])
        time.sleep(0.08)  # Wait 80ms for measurement completion
        
        # Read measurement data
        self._buf = self.bus.read_i2c_block_data(self.address, 0x00, 6 + self.active_crc)

        # Validate CRC if enabled and parse data
        if not self.active_crc or (self._crc8() == self._buf[6]):
            # Parse humidity (20 bits)
            hum = self._buf[1] << 12 | self._buf[2] << 4 | self._buf[3] >> 4
            self.humidity = hum * 100 / 0x100000
            
            # Parse temperature (20 bits)
            temp = (self._buf[3] & 0xF) << 16 | self._buf[4] << 8 | self._buf[5]
            self.temperature = temp * 200.0 / 0x100000 - 50
            return True
        return False