"""
Sensor data preprocessing utilities for LLaVA-Sensing training.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class SensorDataProcessor:
    """
    Process sensor data for training LLaVA-Sensing model.
    """
    
    def __init__(self, 
                 temp_range: Tuple[float, float] = (-20.0, 50.0),
                 humidity_range: Tuple[float, float] = (0.0, 100.0),
                 imu_acc_range: Tuple[float, float] = (-20.0, 20.0),
                 imu_gyro_range: Tuple[float, float] = (-10.0, 10.0)):
        """
        Initialize sensor data processor.
        
        Args:
            temp_range: Temperature normalization range (min, max)
            humidity_range: Humidity normalization range (min, max)  
            imu_acc_range: Accelerometer normalization range (min, max)
            imu_gyro_range: Gyroscope normalization range (min, max)
        """
        self.temp_range = temp_range
        self.humidity_range = humidity_range
        self.imu_acc_range = imu_acc_range
        self.imu_gyro_range = imu_gyro_range
        
    def normalize_temperature(self, temp: float) -> float:
        """Normalize temperature to [-1, 1] range."""
        min_temp, max_temp = self.temp_range
        return 2.0 * (temp - min_temp) / (max_temp - min_temp) - 1.0
    
    def normalize_humidity(self, humidity: float) -> float:
        """Normalize humidity to [-1, 1] range."""
        min_hum, max_hum = self.humidity_range
        return 2.0 * (humidity - min_hum) / (max_hum - min_hum) - 1.0
    
    def normalize_wind_direction(self, wind_direction: float) -> Tuple[float, float]:
        """Convert wind direction to normalized cos/sin components."""
        return np.cos(wind_direction), np.sin(wind_direction)
    
    def normalize_imu(self, imu_data: List[float]) -> List[float]:
        """
        Normalize IMU data (accelerometer + gyroscope).
        
        Args:
            imu_data: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            
        Returns:
            Normalized IMU data
        """
        normalized = []
        
        # Normalize accelerometer (first 3 components)
        acc_min, acc_max = self.imu_acc_range
        for i in range(3):
            normalized.append(2.0 * (imu_data[i] - acc_min) / (acc_max - acc_min) - 1.0)
        
        # Normalize gyroscope (last 3 components)  
        gyro_min, gyro_max = self.imu_gyro_range
        for i in range(3, 6):
            normalized.append(2.0 * (imu_data[i] - gyro_min) / (gyro_max - gyro_min) - 1.0)
            
        return normalized
    
    def process_sensor_data(self, sensor_data: Dict) -> Dict:
        """
        Process raw sensor data into normalized format for model input.
        
        Args:
            sensor_data: Raw sensor data dictionary
            
        Returns:
            Processed sensor data ready for model input
        """
        processed = {}
        
        # Normalize individual sensors
        processed['temperature'] = torch.tensor([[self.normalize_temperature(sensor_data['temperature'])]], 
                                               dtype=torch.float32)
        processed['humidity'] = torch.tensor([[self.normalize_humidity(sensor_data['humidity'])]], 
                                           dtype=torch.float32)
        
        # Wind direction to cos/sin components
        wind_cos, wind_sin = self.normalize_wind_direction(sensor_data['wind_direction'])
        processed['wind_direction'] = torch.tensor([[wind_cos, wind_sin]], dtype=torch.float32)
        
        # Normalize IMU data
        normalized_imu = self.normalize_imu(sensor_data['imu'])
        processed['imu'] = torch.tensor([normalized_imu], dtype=torch.float32)
        
        return processed
    
    def batch_process_sensors(self, sensor_data_list: List[Dict]) -> Dict:
        """
        Process a batch of sensor data.
        
        Args:
            sensor_data_list: List of sensor data dictionaries
            
        Returns:
            Batched and processed sensor data
        """
        batch_size = len(sensor_data_list)
        
        # Initialize batch tensors
        temperature_batch = torch.zeros(batch_size, 1, dtype=torch.float32)
        humidity_batch = torch.zeros(batch_size, 1, dtype=torch.float32)
        wind_batch = torch.zeros(batch_size, 2, dtype=torch.float32)
        imu_batch = torch.zeros(batch_size, 6, dtype=torch.float32)
        
        for i, sensor_data in enumerate(sensor_data_list):
            processed = self.process_sensor_data(sensor_data)
            
            temperature_batch[i] = processed['temperature'].squeeze(0)
            humidity_batch[i] = processed['humidity'].squeeze(0)
            wind_batch[i] = processed['wind_direction'].squeeze(0)
            imu_batch[i] = processed['imu'].squeeze(0)
        
        return {
            'temperature': temperature_batch,
            'humidity': humidity_batch,
            'wind_direction': wind_batch,
            'imu': imu_batch
        }


def load_sensor_literature_data(json_path: str) -> List[Dict]:
    """
    Load sensor-literature dataset from JSON file(s).
    Supports loading multiple chunk files if path contains chunk pattern.
    
    Args:
        json_path: Path to JSON dataset file or chunk pattern
        
    Returns:
        List of dataset entries
    """
    import os
    import glob
    
    # Check if this is a chunk file pattern (e.g., chunk_001.json)
    if 'chunk_' in json_path and json_path.endswith('001.json'):
        # Extract base pattern and load chunks 1-100
        base_path = json_path.replace('001.json', '{:03d}.json')
        dir_path = os.path.dirname(json_path)
        
        all_data = []
        chunk_count = 0
        
        # Load chunks 1-100
        for chunk_num in range(1, 101):
            chunk_file = base_path.format(chunk_num)
            if os.path.exists(chunk_file):
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        all_data.extend(chunk_data)
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            print(f"Loaded chunk {chunk_num}, total entries: {len(all_data)}")
                except Exception as e:
                    print(f"Warning: Could not load chunk {chunk_num}: {e}")
        
        print(f"Successfully loaded {chunk_count} chunks with {len(all_data)} total entries")
        return all_data
    else:
        # Single file loading
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def convert_to_llava_format(dataset_entries: List[Dict]) -> List[Dict]:
    """
    Convert sensor-literature dataset to LLaVA training format.
    
    Args:
        dataset_entries: List of sensor-literature dataset entries
        
    Returns:
        Dataset in LLaVA format
    """
    llava_format = []
    
    for entry in dataset_entries:
        # Create LLaVA-style conversation format
        conversation = {
            "id": entry['id'],
            "conversations": [
                {
                    "from": "human",
                    "value": "Based on the environmental sensor data I'm experiencing, write a literary paragraph describing the scene."
                },
                {
                    "from": "gpt", 
                    "value": entry['target_paragraph']
                }
            ],
            "sensor_data": entry['sensor_data'],  # Add sensor data
            "literary_style": entry.get('literary_style', 'modernist_novel')
        }
        
        llava_format.append(conversation)
    
    return llava_format