import torch
import numpy as np
from typing import Dict, Tuple


class RobotRelativeWindCalculator:
    """
    Calculate wind direction relative to robot's orientation using IMU data.
    """
    
    def __init__(self):
        # Moving average window for IMU filtering
        self.gyro_window_size = 10
        self.acc_window_size = 5
        
        # Historical data for filtering
        self.gyro_history = []
        self.acc_history = []
        
    def filter_imu_data(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Apply filtering to IMU data to reduce noise.
        
        Args:
            imu_data: [batch_size, 6] - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        
        Returns:
            filtered_imu: [batch_size, 6] - filtered IMU data
        """
        # Simple moving average filter (in practice, use Kalman filter)
        if len(self.gyro_history) >= self.gyro_window_size:
            self.gyro_history.pop(0)
        if len(self.acc_history) >= self.acc_window_size:
            self.acc_history.pop(0)
            
        self.gyro_history.append(imu_data[..., 3:6])  # gyro data
        self.acc_history.append(imu_data[..., 0:3])   # acc data
        
        # Average filtering
        if len(self.gyro_history) > 1:
            filtered_gyro = torch.stack(self.gyro_history).mean(dim=0)
        else:
            filtered_gyro = imu_data[..., 3:6]
            
        if len(self.acc_history) > 1:
            filtered_acc = torch.stack(self.acc_history).mean(dim=0)
        else:
            filtered_acc = imu_data[..., 0:3]
            
        return torch.cat([filtered_acc, filtered_gyro], dim=-1)
    
    def estimate_robot_heading(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Estimate robot heading from IMU data.
        
        Args:
            imu_data: [batch_size, 6] - filtered IMU data
        
        Returns:
            heading: [batch_size] - robot heading in radians
        """
        # Extract accelerometer and gyroscope data
        acc = imu_data[..., 0:3]  # [acc_x, acc_y, acc_z]
        gyro = imu_data[..., 3:6]  # [gyro_x, gyro_y, gyro_z]
        
        # Method 1: Use accelerometer to estimate tilt-compensated heading
        # Assuming acc_x points forward, acc_y points right
        heading_from_acc = torch.atan2(acc[..., 1], acc[..., 0])
        
        # Method 2: Integrate gyroscope (simplified - needs proper integration)
        # In practice, you'd maintain state and integrate over time
        heading_from_gyro = torch.atan2(gyro[..., 1], gyro[..., 0])
        
        # Fusion of both methods (weighted average)
        # In practice, use proper sensor fusion like Complementary or Kalman filter
        alpha = 0.7  # weight for accelerometer
        heading = alpha * heading_from_acc + (1 - alpha) * heading_from_gyro
        
        return heading
    
    def calculate_relative_wind_direction(self, 
                                        wind_direction: torch.Tensor, 
                                        imu_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate wind direction relative to robot's orientation.
        
        Args:
            wind_direction: [batch_size, 1] - global wind direction in radians
            imu_data: [batch_size, 6] - IMU sensor data
        
        Returns:
            dict containing:
                - relative_angle: [batch_size] - relative wind angle in radians
                - relative_vector: [batch_size, 2] - relative wind unit vector [cos, sin]
                - literary_desc: list of literary descriptions
        """
        # Filter IMU data
        filtered_imu = self.filter_imu_data(imu_data)
        
        # Estimate robot heading
        robot_heading = self.estimate_robot_heading(filtered_imu)
        
        # Calculate relative wind direction
        relative_angle = wind_direction.squeeze(-1) - robot_heading
        
        # Normalize angle to [-π, π]
        relative_angle = torch.atan2(torch.sin(relative_angle), torch.cos(relative_angle))
        
        # Convert to unit vector
        relative_vector = torch.stack([
            torch.cos(relative_angle),
            torch.sin(relative_angle)
        ], dim=-1)
        
        # Generate literary descriptions
        literary_descriptions = []
        for i in range(relative_angle.shape[0]):
            angle_deg = torch.rad2deg(relative_angle[i]).item()
            desc = self.get_literary_wind_description(angle_deg)
            literary_descriptions.append(desc)
        
        return {
            'relative_angle': relative_angle,
            'relative_vector': relative_vector,
            'literary_descriptions': literary_descriptions,
            'robot_heading': robot_heading,
            'global_wind_direction': wind_direction.squeeze(-1)
        }
    
    def get_literary_wind_description(self, relative_angle_deg: float) -> Dict[str, str]:
        """
        Generate literary description based on relative wind angle.
        
        Args:
            relative_angle_deg: Relative wind angle in degrees
        
        Returns:
            dict: Literary descriptors for the wind direction
        """
        # Normalize angle to [0, 360)
        angle = relative_angle_deg
        if angle < 0:
            angle += 360
            
        if -22.5 <= angle < 22.5 or 337.5 <= angle <= 360:
            return {
                "direction": "정면에서",
                "sensation": "마주치는",
                "korean_desc": "정면으로 불어오는 바람이 얼굴을 스치며",
                "poetic_desc": "앞에서 불어오는 바람이 이마를 어루만지며 지나간다",
                "intensity": "강한 대면감"
            }
        elif 22.5 <= angle < 67.5:
            return {
                "direction": "오른쪽 비스듬히",
                "sensation": "스며드는",
                "korean_desc": "오른쪽으로 기울어진 바람이 어깨를 감싸며",
                "poetic_desc": "오른편에서 비스듬히 스며드는 바람이 목덜미를 스친다",
                "intensity": "부드러운 접촉감"
            }
        elif 67.5 <= angle < 112.5:
            return {
                "direction": "오른쪽에서",
                "sensation": "스치는",
                "korean_desc": "오른쪽에서 불어오는 바람이 몸을 스치며",
                "poetic_desc": "오른쪽에서 불어오는 바람이 어깨를 타고 흘러간다",
                "intensity": "측면의 흐름감"
            }
        elif 112.5 <= angle < 157.5:
            return {
                "direction": "오른쪽 뒤에서",
                "sensation": "밀어주는",
                "korean_desc": "뒤에서 밀어주는 바람이 등을 어루만지며",
                "poetic_desc": "오른쪽 뒤에서 불어오는 바람이 등을 가볍게 떠밀어준다",
                "intensity": "지지하는 힘"
            }
        elif 157.5 <= angle < 202.5:
            return {
                "direction": "뒤에서",
                "sensation": "떠미는",
                "korean_desc": "뒤에서 불어오는 바람이 등을 밀어내며",
                "poetic_desc": "뒤에서 불어오는 바람이 등을 두드리며 앞으로 나아가게 한다",
                "intensity": "추진하는 힘"
            }
        elif 202.5 <= angle < 247.5:
            return {
                "direction": "왼쪽 뒤에서",
                "sensation": "감싸는",
                "korean_desc": "왼쪽 뒤에서 불어오는 바람이 어깨를 감싸며",
                "poetic_desc": "왼쪽 뒤편에서 불어오는 바람이 등줄기를 따라 흐른다",
                "intensity": "포근한 감쌈"
            }
        elif 247.5 <= angle < 292.5:
            return {
                "direction": "왼쪽에서",
                "sensation": "스치는",
                "korean_desc": "왼쪽에서 불어오는 바람이 몸을 스치며",
                "poetic_desc": "왼쪽에서 불어오는 바람이 옆구리를 훑고 지나간다",
                "intensity": "측면의 흐름감"
            }
        else:  # 292.5 <= angle < 337.5
            return {
                "direction": "왼쪽 비스듬히",
                "sensation": "스며드는",
                "korean_desc": "왼쪽으로 기울어진 바람이 어깨를 감싸며",
                "poetic_desc": "왼편에서 비스듬히 불어오는 바람이 뺨을 간질인다",
                "intensity": "부드러운 접촉감"
            }


class EnvironmentalSensorProcessor:
    """
    Process all environmental sensor data for literary generation.
    """
    
    def __init__(self):
        self.wind_calculator = RobotRelativeWindCalculator()
        
    def process_sensors(self, sensor_data: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """
        Process all sensor data and generate literary context.
        
        Args:
            sensor_data: Dict containing temperature, humidity, wind_direction, imu
        
        Returns:
            dict: Processed sensor data with literary context
        """
        # Calculate relative wind
        wind_info = self.wind_calculator.calculate_relative_wind_direction(
            sensor_data['wind_direction'], 
            sensor_data['imu']
        )
        
        # Process temperature for literary context
        temp_celsius = sensor_data['temperature'].squeeze(-1)
        temp_desc = self.get_temperature_description(temp_celsius)
        
        # Process humidity for literary context  
        humidity_percent = sensor_data['humidity'].squeeze(-1)
        humidity_desc = self.get_humidity_description(humidity_percent)
        
        return {
            'raw_sensors': sensor_data,
            'wind': wind_info,
            'temperature': {
                'value': temp_celsius,
                'description': temp_desc
            },
            'humidity': {
                'value': humidity_percent, 
                'description': humidity_desc
            },
            'literary_context': self.generate_literary_context(
                temp_desc, humidity_desc, wind_info['literary_descriptions'][0]
            )
        }
    
    def get_temperature_description(self, temp_celsius: torch.Tensor) -> Dict[str, str]:
        """Generate literary description for temperature."""
        temp = temp_celsius.mean().item()  # Average across batch
        
        if temp < 0:
            return {
                "sensation": "얼어붙는", 
                "description": "뼛속까지 파고드는 차가운 공기",
                "mood": "냉엄한"
            }
        elif 0 <= temp < 10:
            return {
                "sensation": "차가운",
                "description": "피부를 에는 듯한 차가운 기운", 
                "mood": "서늘한"
            }
        elif 10 <= temp < 20:
            return {
                "sensation": "시원한",
                "description": "상쾌한 기운이 감도는 공기",
                "mood": "상쾌한"
            }
        elif 20 <= temp < 30:
            return {
                "sensation": "따뜻한", 
                "description": "살갗을 부드럽게 감싸는 온기",
                "mood": "포근한"
            }
        else:
            return {
                "sensation": "뜨거운",
                "description": "숨이 막힐 듯한 무더운 열기",
                "mood": "후텁지근한"
            }
    
    def get_humidity_description(self, humidity_percent: torch.Tensor) -> Dict[str, str]:
        """Generate literary description for humidity."""
        humidity = humidity_percent.mean().item()
        
        if humidity < 30:
            return {
                "sensation": "건조한",
                "description": "메마른 공기가 목을 칼칼하게 만들며",
                "mood": "메마른"
            }
        elif 30 <= humidity < 60:
            return {
                "sensation": "적당한", 
                "description": "기분 좋게 균형 잡힌 공기가",
                "mood": "쾌적한"
            }
        elif 60 <= humidity < 80:
            return {
                "sensation": "촉촉한",
                "description": "살짝 촉촉한 기운이 감도는 공기가",
                "mood": "부드러운"
            }
        else:
            return {
                "sensation": "끈적한",
                "description": "끈적끈적한 습기가 온몸을 감싸며",
                "mood": "무겁고 축축한"
            }
    
    def generate_literary_context(self, temp_desc: Dict, humidity_desc: Dict, wind_desc: Dict) -> str:
        """Generate combined literary context from all sensors."""
        return f"{temp_desc['description']} {humidity_desc['description']} {wind_desc['poetic_desc']}"