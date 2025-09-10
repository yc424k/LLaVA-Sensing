import torch
import torch.nn as nn
import numpy as np


class EnvironmentalSensorEncoder(nn.Module):
    """
    Environmental sensor encoder for literary paragraph generation.
    Handles temperature, humidity, wind direction, and IMU data.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.sensor_embed_dim = getattr(config, 'sensor_embed_dim', 256)
        self.hidden_size = getattr(config, 'hidden_size', 4096)
        
        # Individual sensor encoders
        self.temp_embed = nn.Sequential(
            nn.Linear(1, self.sensor_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.sensor_embed_dim // 2, self.sensor_embed_dim)
        )
        
        self.humidity_embed = nn.Sequential(
            nn.Linear(1, self.sensor_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.sensor_embed_dim // 2, self.sensor_embed_dim)
        )
        
        # Wind direction encoder (direction only, no speed)
        # Input: [cos(wind_angle), sin(wind_angle)] - normalized direction vector
        self.wind_embed = nn.Sequential(
            nn.Linear(2, self.sensor_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.sensor_embed_dim // 2, self.sensor_embed_dim)
        )
        
        # IMU encoder (accelerometer + gyroscope)
        # Input: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        self.imu_embed = nn.Sequential(
            nn.Linear(6, self.sensor_embed_dim),
            nn.ReLU(),
            nn.Linear(self.sensor_embed_dim, self.sensor_embed_dim)
        )
        
        # Robot-relative wind direction encoder
        # Input: relative wind direction from robot's perspective
        self.relative_wind_embed = nn.Sequential(
            nn.Linear(2, self.sensor_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.sensor_embed_dim // 2, self.sensor_embed_dim)
        )
        
        # Cross-modal attention for sensor fusion
        self.sensor_attention = nn.MultiheadAttention(
            embed_dim=self.sensor_embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final projection to match language model hidden size
        self.final_projection = nn.Linear(
            self.sensor_embed_dim * 5,  # 5 sensor modalities
            self.hidden_size
        )
        
        # Positional encoding for temporal sensor data
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 5, self.sensor_embed_dim)
        )
        
    def compute_robot_relative_wind(self, wind_direction, imu_data):
        """
        Compute wind direction relative to robot's orientation.
        
        Args:
            wind_direction: Global wind direction in radians
            imu_data: IMU sensor data [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        
        Returns:
            relative_wind: Wind direction relative to robot [cos, sin]
        """
        # Extract robot heading from IMU (simplified - assumes yaw from gyro_z integration)
        # In practice, you'd use proper IMU fusion algorithms
        robot_heading = torch.atan2(imu_data[..., 4], imu_data[..., 3])  # gyro_y, gyro_x
        
        # Calculate relative wind direction
        relative_angle = wind_direction - robot_heading
        
        # Convert to unit vector
        relative_wind = torch.stack([
            torch.cos(relative_angle),
            torch.sin(relative_angle)
        ], dim=-1)
        
        return relative_wind
        
    def forward(self, sensor_data):
        """
        Forward pass for environmental sensor encoding.
        
        Args:
            sensor_data: Dict containing:
                - temperature: [batch_size, 1] - temperature in Celsius
                - humidity: [batch_size, 1] - relative humidity (0-100)
                - wind_direction: [batch_size, 1] - wind direction in radians
                - imu: [batch_size, 6] - IMU data
        
        Returns:
            sensor_features: [batch_size, 1, hidden_size] - encoded sensor features
        """
        batch_size = sensor_data['temperature'].shape[0]
        
        # Encode individual sensors
        temp_features = self.temp_embed(sensor_data['temperature'])  # [B, embed_dim]
        humidity_features = self.humidity_embed(sensor_data['humidity'])  # [B, embed_dim]
        
        # Wind direction is already in [cos, sin] format from preprocessing
        wind_vector = sensor_data['wind_direction']  # [B, 2] - already [cos, sin]
        wind_features = self.wind_embed(wind_vector)  # [B, embed_dim]
        
        # Encode IMU data
        imu_features = self.imu_embed(sensor_data['imu'])  # [B, embed_dim]
        
        # Compute robot-relative wind direction
        # Convert wind_vector [cos, sin] back to angle for relative calculation
        wind_angle = torch.atan2(wind_vector[:, 1], wind_vector[:, 0])  # [B]
        relative_wind = self.compute_robot_relative_wind(wind_angle, sensor_data['imu'])
        relative_wind_features = self.relative_wind_embed(relative_wind)  # [B, embed_dim]
        
        # Stack all sensor features
        all_features = torch.stack([
            temp_features,
            humidity_features, 
            wind_features,
            imu_features,
            relative_wind_features
        ], dim=1)  # [B, 5, embed_dim]
        
        # Add positional encoding
        all_features = all_features + self.pos_encoding
        
        # Apply cross-modal attention
        attended_features, _ = self.sensor_attention(
            all_features, all_features, all_features
        )  # [B, 5, embed_dim]
        
        # Flatten and project to final hidden size
        flattened = attended_features.reshape(batch_size, -1)  # [B, 5 * embed_dim]
        final_features = self.final_projection(flattened)  # [B, hidden_size]
        
        # Add sequence dimension for compatibility with language model
        return final_features.unsqueeze(1)  # [B, 1, hidden_size]
        
    @property
    def dummy_feature(self):
        """Dummy feature for initialization."""
        return torch.zeros(1, self.hidden_size, device=self.pos_encoding.device)
        
    @property
    def dtype(self):
        return self.pos_encoding.dtype
        
    @property
    def device(self):
        return self.pos_encoding.device


def create_sensor_data_sample():
    """
    Create sample sensor data for testing.
    """
    return {
        'temperature': torch.tensor([[25.5]], dtype=torch.float32),  # 25.5°C
        'humidity': torch.tensor([[60.0]], dtype=torch.float32),     # 60% RH
        'wind_direction': torch.tensor([[np.pi/4]], dtype=torch.float32),  # 45° (NE)
        'imu': torch.tensor([[0.1, 0.05, 9.8, 0.01, 0.02, 0.1]], dtype=torch.float32)  # IMU data
    }


def interpret_relative_wind_direction(relative_wind_vector):
    """
    Interpret relative wind direction for literary description.
    
    Args:
        relative_wind_vector: [cos, sin] of relative wind direction
    
    Returns:
        dict: Literary descriptors for wind direction
    """
    angle = torch.atan2(relative_wind_vector[1], relative_wind_vector[0])
    angle_degrees = torch.rad2deg(angle).item()
    
    # Normalize to [0, 360)
    if angle_degrees < 0:
        angle_degrees += 360
        
    # Literary descriptions based on relative direction
    if -22.5 <= angle_degrees < 22.5 or 337.5 <= angle_degrees < 360:
        return {
            "direction": "from ahead",
            "sensation": "confronting", 
            "description": "the wind blowing straight ahead brushes against the face"
        }
    elif 22.5 <= angle_degrees < 67.5:
        return {
            "direction": "from the right front",
            "sensation": "slanting and penetrating",
            "description": "the wind tilting to the right embraces the shoulder"
        }
    elif 67.5 <= angle_degrees < 112.5:
        return {
            "direction": "from the right", 
            "sensation": "brushing sideways",
            "description": "the wind blowing from the right brushes against the body"
        }
    elif 112.5 <= angle_degrees < 157.5:
        return {
            "direction": "from the right rear",
            "sensation": "following behind", 
            "description": "the wind pushing from behind caresses the back"
        }
    elif 157.5 <= angle_degrees < 202.5:
        return {
            "direction": "from behind",
            "sensation": "pushing",
            "description": "the wind blowing from behind pushes against the back"
        }
    elif 202.5 <= angle_degrees < 247.5:
        return {
            "direction": "from the left rear", 
            "sensation": "following behind",
            "description": "the wind blowing from the left rear embraces the shoulder"
        }
    elif 247.5 <= angle_degrees < 292.5:
        return {
            "direction": "from the left",
            "sensation": "brushing sideways", 
            "description": "the wind blowing from the left brushes against the body"
        }
    else:  # 292.5 <= angle_degrees < 337.5
        return {
            "direction": "from the left front",
            "sensation": "slanting and penetrating",
            "description": "the wind tilting to the left embraces the shoulder"
        }