"""
Advanced corpus processor for literary text analysis and sensor data mapping.
"""

import json
import re
import pandas as pd
from typing import Dict, List, Tuple, Set
import numpy as np
from pathlib import Path
import logging
from collections import Counter, defaultdict
import sqlite3
from dataclasses import dataclass, asdict


@dataclass
class LiteraryStyle:
    """Literary style classification."""
    name: str
    characteristics: List[str]
    keywords: List[str]
    sentence_patterns: List[str]


class LiteraryStyleClassifier:
    """
    Classify literary passages by style (modernism, travel writing, etc.)
    """
    
    def __init__(self):
        self.styles = {
            "modernism": LiteraryStyle(
                name="modernism",
                characteristics=["stream_of_consciousness", "interior_monologue", "nonlinear_time"],
                keywords=["memory", "consciousness", "time", "moment", "feeling", "thought"],
                sentence_patterns=[r".*\.{3}", r".*\?.*\?", r"he\s+thought"]
            ),
            "naturalism": LiteraryStyle(
                name="naturalism", 
                characteristics=["environmental_description", "objective_observation", "scientific_accuracy"],
                keywords=["nature", "environment", "wind", "temperature", "humidity", "air"],
                sentence_patterns=[r".*was\s+", r".*it\s+is"]
            ),
            "travel_writing": LiteraryStyle(
                name="travel_writing",
                characteristics=["place_description", "movement_experience", "cultural_observation"],
                keywords=["path", "walk", "journey", "place", "landscape", "experience"],
                sentence_patterns=[r".*from.*to", r".*through.*"]
            )
        }
        
    def classify_passage(self, text: str) -> Dict[str, float]:
        """
        Classify literary passage by style.
        
        Args:
            text: Literary text passage
            
        Returns:
            dict: Style probabilities
        """
        scores = {}
        text_lower = text.lower()
        
        for style_name, style in self.styles.items():
            score = 0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in style.keywords if keyword in text_lower)
            score += keyword_matches / len(style.keywords)
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in style.sentence_patterns if re.search(pattern, text))
            score += pattern_matches / len(style.sentence_patterns) * 0.5
            
            scores[style_name] = score
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        return scores


class SensorContextExtractor:
    """
    Extract detailed sensor context from literary passages.
    """
    
    def __init__(self):
        self.setup_patterns()
        
    def setup_patterns(self):
        """Setup regex patterns for context extraction."""
        
        # Temperature patterns
        self.temp_patterns = {
            'hot_extreme': [r'scorching', r'blazing', r'sweltering', r'burning'],
            'hot_mild': [r'warm', r'mild', r'balmy', r'tepid'],
            'cold_extreme': [r'freezing', r'frigid', r'icy', r'arctic'],
            'cold_mild': [r'cool', r'crisp', r'chilly', r'fresh']
        }
        
        # Humidity patterns  
        self.humidity_patterns = {
            'very_humid': [r'sticky', r'clammy', r'muggy'],
            'humid': [r'moist', r'damp', r'dewy'],
            'dry': [r'dry', r'arid', r'parched'],
            'very_dry': [r'desiccated', r'withered', r'dusty']
        }
        
        # Wind patterns with direction
        self.wind_patterns = {
            'front': [r'front.*wind', r'ahead.*wind', r'facing.*wind'],
            'back': [r'behind.*wind', r'back.*wind', r'pushing.*wind'],
            'side': [r'side.*wind', r'lateral.*wind', r'cross.*wind'],
            'gentle': [r'gentle.*breeze', r'soft.*wind', r'light.*breeze'],
            'strong': [r'strong.*wind', r'fierce.*wind', r'gale']
        }
        
        # Movement patterns
        self.movement_patterns = {
            'walking': [r'walk', r'stroll', r'pace', r'footstep'],
            'running': [r'run', r'sprint', r'dash', r'race'],
            'standing': [r'stand', r'stop', r'pause', r'still'],
            'climbing': [r'climb', r'ascend', r'scale', r'clamber']
        }
        
        # Time patterns
        self.time_patterns = {
            'dawn': [r'dawn', r'daybreak', r'sunrise'],
            'morning': [r'morning', r'am', r'forenoon'],
            'noon': [r'noon', r'midday', r'day'],
            'afternoon': [r'afternoon', r'pm', r'evening'],
            'evening': [r'evening', r'sunset', r'dusk'],
            'night': [r'night', r'midnight', r'darkness']
        }
    
    def extract_sensor_context(self, text: str) -> Dict:
        """
        Extract comprehensive sensor context from text.
        
        Args:
            text: Literary text passage
            
        Returns:
            dict: Extracted sensor context
        """
        context = {
            'temperature': self._extract_temperature_context(text),
            'humidity': self._extract_humidity_context(text),
            'wind': self._extract_wind_context(text),
            'movement': self._extract_movement_context(text),
            'time': self._extract_time_context(text),
            'weather': self._extract_weather_context(text),
            'location': self._extract_location_context(text)
        }
        
        return context
    
    def _extract_temperature_context(self, text: str) -> Dict:
        """Extract temperature-related context."""
        temp_context = {'level': 'neutral', 'intensity': 0, 'keywords': []}
        
        for level, patterns in self.temp_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    temp_context['keywords'].extend(matches)
                    if 'hot' in level:
                        temp_context['level'] = 'hot'
                        temp_context['intensity'] = 2 if 'extreme' in level else 1
                    elif 'cold' in level:
                        temp_context['level'] = 'cold' 
                        temp_context['intensity'] = 2 if 'extreme' in level else 1
        
        return temp_context
    
    def _extract_humidity_context(self, text: str) -> Dict:
        """Extract humidity-related context."""
        humidity_context = {'level': 'neutral', 'intensity': 0, 'keywords': []}
        
        for level, patterns in self.humidity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    humidity_context['keywords'].extend(matches)
                    if 'humid' in level:
                        humidity_context['level'] = 'humid'
                        humidity_context['intensity'] = 2 if 'very' in level else 1
                    elif 'dry' in level:
                        humidity_context['level'] = 'dry'
                        humidity_context['intensity'] = 2 if 'very' in level else 1
        
        return humidity_context
    
    def _extract_wind_context(self, text: str) -> Dict:
        """Extract wind-related context."""
        wind_context = {'direction': 'unknown', 'strength': 'mild', 'keywords': []}
        
        for direction, patterns in self.wind_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    wind_context['keywords'].extend(matches)
                    if direction in ['front', 'back', 'side']:
                        wind_context['direction'] = direction
                    elif direction in ['gentle', 'strong']:
                        wind_context['strength'] = direction
        
        return wind_context
    
    def _extract_movement_context(self, text: str) -> Dict:
        """Extract movement-related context."""
        movement_context = {'type': 'stationary', 'intensity': 0, 'keywords': []}
        
        for movement_type, patterns in self.movement_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    movement_context['keywords'].extend(matches)
                    movement_context['type'] = movement_type
                    if movement_type == 'running':
                        movement_context['intensity'] = 3
                    elif movement_type in ['walking', 'climbing']:
                        movement_context['intensity'] = 2
                    else:
                        movement_context['intensity'] = 1
        
        return movement_context
    
    def _extract_time_context(self, text: str) -> Dict:
        """Extract time-related context."""
        time_context = {'period': 'unknown', 'keywords': []}
        
        for period, patterns in self.time_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    time_context['keywords'].extend(matches)
                    time_context['period'] = period
                    break
        
        return time_context
    
    def _extract_weather_context(self, text: str) -> Dict:
        """Extract weather-related context."""
        weather_patterns = {
            'rain': [r'rain', r'drizzle', r'shower', r'downpour'],
            'snow': [r'snow', r'snowfall', r'flake'],
            'fog': [r'fog', r'mist', r'haze'],
            'clear': [r'clear', r'bright', r'sunny'],
            'cloudy': [r'cloudy', r'overcast', r'grey']
        }
        
        weather_context = {'condition': 'unknown', 'keywords': []}
        
        for condition, patterns in weather_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    weather_context['keywords'].extend(matches)
                    weather_context['condition'] = condition
                    break
        
        return weather_context
    
    def _extract_location_context(self, text: str) -> Dict:
        """Extract location-related context."""
        location_patterns = {
            'urban': [r'city', r'street', r'building', r'asphalt', r'traffic'],
            'forest': [r'forest', r'trees', r'leaves', r'woods'],
            'mountain': [r'mountain', r'peak', r'ridge', r'rock'],
            'beach': [r'ocean', r'beach', r'waves', r'sand'],
            'park': [r'park', r'grass', r'bench', r'garden'],
            'indoor': [r'indoor', r'room', r'house', r'inside']
        }
        
        location_context = {'type': 'unknown', 'keywords': []}
        
        for location_type, patterns in location_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    location_context['keywords'].extend(matches)
                    location_context['type'] = location_type
                    break
        
        return location_context


class AdvancedSensorMapper:
    """
    Map extracted literary context to realistic sensor values.
    """
    
    def __init__(self):
        self.temp_mappings = {
            ('hot', 2): (30, 40, 5),      # (mean, max, std)
            ('hot', 1): (25, 32, 3),
            ('cold', 2): (-5, 10, 8),
            ('cold', 1): (10, 20, 5),
            ('neutral', 0): (15, 25, 8)
        }
        
        self.humidity_mappings = {
            ('humid', 2): (80, 100, 10),
            ('humid', 1): (60, 85, 12),
            ('dry', 2): (10, 30, 8),
            ('dry', 1): (30, 50, 10),
            ('neutral', 0): (40, 70, 15)
        }
        
        self.wind_direction_mappings = {
            'front': (0, 0.3),            # (base_angle, noise)
            'back': (np.pi, 0.3),
            'side': (np.pi/2, 0.5),       # Could be either side
            'unknown': (0, 2*np.pi)       # Uniform random
        }
        
        self.movement_imu_mappings = {
            ('running', 3): {'acc_std': 3.0, 'gyro_std': 0.5},
            ('walking', 2): {'acc_std': 1.5, 'gyro_std': 0.2},
            ('climbing', 2): {'acc_std': 2.0, 'gyro_std': 0.3},
            ('standing', 1): {'acc_std': 0.5, 'gyro_std': 0.1},
            ('stationary', 0): {'acc_std': 0.2, 'gyro_std': 0.05}
        }
    
    def map_to_sensor_values(self, context: Dict) -> Dict:
        """
        Map literary context to sensor values.
        
        Args:
            context: Extracted sensor context
            
        Returns:
            dict: Mapped sensor values
        """
        sensor_data = {}
        
        # Map temperature
        temp_key = (context['temperature']['level'], context['temperature']['intensity'])
        if temp_key in self.temp_mappings:
            mean, max_val, std = self.temp_mappings[temp_key]
            temperature = np.random.normal(mean, std)
            temperature = max(-20, min(max_val, temperature))
        else:
            temperature = np.random.normal(20, 10)
        
        sensor_data['temperature'] = round(temperature, 1)
        
        # Map humidity
        humidity_key = (context['humidity']['level'], context['humidity']['intensity'])
        if humidity_key in self.humidity_mappings:
            mean, max_val, std = self.humidity_mappings[humidity_key]
            humidity = np.random.normal(mean, std)
            humidity = max(0, min(max_val, humidity))
        else:
            humidity = np.random.normal(50, 20)
        
        sensor_data['humidity'] = round(max(0, min(100, humidity)), 1)
        
        # Map wind direction
        wind_direction = context['wind']['direction']
        if wind_direction in self.wind_direction_mappings:
            base_angle, noise = self.wind_direction_mappings[wind_direction]
            if noise >= 2*np.pi:  # Uniform case
                wind_angle = np.random.uniform(0, 2*np.pi)
            else:
                wind_angle = base_angle + np.random.normal(0, noise)
        else:
            wind_angle = np.random.uniform(0, 2*np.pi)
        
        sensor_data['wind_direction'] = round(wind_angle % (2*np.pi), 3)
        
        # Map IMU data
        movement_key = (context['movement']['type'], context['movement']['intensity'])
        if movement_key in self.movement_imu_mappings:
            imu_params = self.movement_imu_mappings[movement_key]
        else:
            imu_params = self.movement_imu_mappings[('stationary', 0)]
        
        acc_x = np.random.normal(0, imu_params['acc_std'])
        acc_y = np.random.normal(0, imu_params['acc_std'])
        acc_z = 9.8 + np.random.normal(0, imu_params['acc_std'] * 0.2)
        
        gyro_x = np.random.normal(0, imu_params['gyro_std'])
        gyro_y = np.random.normal(0, imu_params['gyro_std'])
        gyro_z = np.random.normal(0, imu_params['gyro_std'] * 0.5)
        
        sensor_data['imu'] = [
            round(acc_x, 3), round(acc_y, 3), round(acc_z, 3),
            round(gyro_x, 3), round(gyro_y, 3), round(gyro_z, 3)
        ]
        
        # Map context
        scenario_mapping = {
            'urban': 'city_walking',
            'forest': 'forest_exploration',
            'mountain': 'mountain_climbing',
            'beach': 'beach_walking',
            'park': 'park_stroll'
        }
        
        time_mapping = {
            'dawn': 'dawn', 'morning': 'morning', 'noon': 'noon',
            'afternoon': 'afternoon', 'evening': 'evening', 'night': 'night'
        }
        
        weather_mapping = {
            'rain': 'rain', 'snow': 'snow', 'clear': 'clear',
            'cloudy': 'cloudy', 'fog': 'fog'
        }
        
        sensor_data['context'] = {
            'scenario': scenario_mapping.get(context['location']['type'], 'city_walking'),
            'time': time_mapping.get(context['time']['period'], 'afternoon'),
            'weather': weather_mapping.get(context['weather']['condition'], 'clear')
        }
        
        return sensor_data
    
    def calculate_mapping_confidence(self, context: Dict) -> float:
        """Calculate confidence score for sensor mapping."""
        confidence_factors = []
        
        # Temperature confidence
        if context['temperature']['keywords']:
            confidence_factors.append(min(1.0, len(context['temperature']['keywords']) * 0.3))
        else:
            confidence_factors.append(0.2)
        
        # Humidity confidence
        if context['humidity']['keywords']:
            confidence_factors.append(min(1.0, len(context['humidity']['keywords']) * 0.3))
        else:
            confidence_factors.append(0.2)
        
        # Wind confidence
        if context['wind']['keywords']:
            confidence_factors.append(min(1.0, len(context['wind']['keywords']) * 0.4))
        else:
            confidence_factors.append(0.1)
        
        # Movement confidence
        if context['movement']['keywords']:
            confidence_factors.append(min(1.0, len(context['movement']['keywords']) * 0.3))
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)


def main():
    """Example usage of advanced corpus processor."""
    
    # Initialize components
    style_classifier = LiteraryStyleClassifier()
    context_extractor = SensorContextExtractor()
    sensor_mapper = AdvancedSensorMapper()
    
    # Example text
    example_text = """
    The dawn air felt cold against his face, as a front wind 
    brushed past him. He walked slowly through the forest path, 
    observing the moist dew-covered leaves.
    """
    
    # Process text
    print("=== Literary Style Classification ===")
    styles = style_classifier.classify_passage(example_text)
    for style, score in styles.items():
        print(f"{style}: {score:.3f}")
    
    print("\n=== Sensor Context Extraction ===")
    context = context_extractor.extract_sensor_context(example_text)
    for key, value in context.items():
        print(f"{key}: {value}")
    
    print("\n=== Sensor Value Mapping ===")
    sensor_data = sensor_mapper.map_to_sensor_values(context)
    confidence = sensor_mapper.calculate_mapping_confidence(context)
    
    print(f"Temperature: {sensor_data['temperature']}°C")
    print(f"Humidity: {sensor_data['humidity']}%")
    print(f"Wind Direction: {sensor_data['wind_direction']} rad")
    print(f"IMU: {sensor_data['imu']}")
    print(f"Context: {sensor_data['context']}")
    print(f"Mapping Confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()