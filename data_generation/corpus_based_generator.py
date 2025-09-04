"""
Corpus-based dataset generator that creates sensor-literature pairs 
from existing literary texts.
"""

import json
import re
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import logging
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SensorKeywords:
    """Keywords that indicate specific sensor conditions."""
    temperature_hot: List[str]
    temperature_cold: List[str] 
    humidity_dry: List[str]
    humidity_wet: List[str]
    wind_keywords: List[str]
    movement_keywords: List[str]
    sensory_keywords: List[str]


class CorpusAnalyzer:
    """
    Analyze literary corpus to extract sensor-relevant passages.
    """
    
    def __init__(self):
        self.sensor_keywords = SensorKeywords(
            temperature_hot=["hot", "scorching", "burning", "blazing", "sweltering", "sultry", "warm", "mild", "tepid", "balmy", "toasty", "heated", "fiery", "boiling", "steaming"],
            temperature_cold=["cold", "freezing", "frigid", "icy", "chilly", "cool", "crisp", "frosty", "bitter", "arctic", "frozen", "numb", "shivering", "brisk"],
            humidity_dry=["dry", "arid", "parched", "withered", "dusty", "crisp", "brittle", "desiccated", "drought", "thirsty"],
            humidity_wet=["humid", "moist", "damp", "wet", "soggy", "sticky", "clammy", "muggy", "steamy", "dewy", "misty", "saturated"],
            wind_keywords=["wind", "breeze", "gale", "gust", "draft", "zephyr", "gentle", "fierce", "howling", "whistling", "rustling", "swirling", "billowing", "fluttering"],
            movement_keywords=["walk", "stride", "march", "run", "sprint", "dash", "move", "step", "pace", "footstep", "motion", "vibration", "shake", "tremble"],
            sensory_keywords=["feel", "sense", "touch", "texture", "skin", "body", "face", "hand", "foot", "breath", "smell", "scent", "aroma", "fragrance"]
        )
        
        # Setup NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.logger = logging.getLogger(__name__)
    
    def load_corpus(self, corpus_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        Load literary corpus from file(s).
        
        Args:
            corpus_path: Path to corpus file or directory
            encoding: File encoding
            
        Returns:
            list: List of text passages
        """
        texts = []
        
        if os.path.isfile(corpus_path):
            # Single file
            texts.extend(self._load_single_file(corpus_path, encoding))
        elif os.path.isdir(corpus_path):
            # Directory of files
            for filename in os.listdir(corpus_path):
                if filename.endswith(('.txt', '.md', '.json')):
                    file_path = os.path.join(corpus_path, filename)
                    texts.extend(self._load_single_file(file_path, encoding))
        
        self.logger.info(f"Loaded {len(texts)} text passages from corpus")
        return texts
    
    def _load_single_file(self, file_path: str, encoding: str) -> List[str]:
        """Load single file and split into passages."""
        texts = []
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            if file_path.endswith('.json'):
                # Handle JSON format
                data = json.loads(content)
                if isinstance(data, list):
                    texts.extend([str(item) for item in data])
                elif isinstance(data, dict):
                    # Extract text fields
                    for key in ['text', 'content', 'body', 'paragraph']:
                        if key in data:
                            if isinstance(data[key], list):
                                texts.extend(data[key])
                            else:
                                texts.append(data[key])
            else:
                # Plain text - split by paragraphs or sentences
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if len(paragraph) > 50:  # Minimum length
                        texts.append(paragraph)
        
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
        
        return texts
    
    def extract_sensory_passages(self, texts: List[str], min_length: int = 100) -> List[Dict]:
        """
        Extract passages that contain sensory descriptions.
        
        Args:
            texts: List of text passages
            min_length: Minimum passage length
            
        Returns:
            list: Sensory passages with metadata
        """
        sensory_passages = []
        
        for i, text in enumerate(texts):
            if len(text) < min_length:
                continue
            
            # Analyze sensory content
            sensory_analysis = self.analyze_sensory_content(text)
            
            # Keep passages with significant sensory content
            if sensory_analysis['total_sensory_score'] > 0.3:
                passage = {
                    'id': f'corpus_passage_{i:06d}',
                    'text': text,
                    'sensory_analysis': sensory_analysis,
                    'length': len(text),
                    'sentences': len(sent_tokenize(text))
                }
                sensory_passages.append(passage)
        
        self.logger.info(f"Extracted {len(sensory_passages)} sensory passages")
        return sensory_passages
    
    def analyze_sensory_content(self, text: str) -> Dict:
        """
        Analyze sensory content in text passage.
        
        Args:
            text: Text passage to analyze
            
        Returns:
            dict: Sensory content analysis
        """
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        word_count = len(words)
        
        # Count sensory keywords
        temp_hot_count = sum(1 for word in self.sensor_keywords.temperature_hot if word in text_lower)
        temp_cold_count = sum(1 for word in self.sensor_keywords.temperature_cold if word in text_lower)
        humidity_dry_count = sum(1 for word in self.sensor_keywords.humidity_dry if word in text_lower)
        humidity_wet_count = sum(1 for word in self.sensor_keywords.humidity_wet if word in text_lower)
        wind_count = sum(1 for word in self.sensor_keywords.wind_keywords if word in text_lower)
        movement_count = sum(1 for word in self.sensor_keywords.movement_keywords if word in text_lower)
        sensory_count = sum(1 for word in self.sensor_keywords.sensory_keywords if word in text_lower)
        
        total_sensory = temp_hot_count + temp_cold_count + humidity_dry_count + humidity_wet_count + wind_count + movement_count + sensory_count
        
        analysis = {
            'temperature_hot_score': temp_hot_count / word_count,
            'temperature_cold_score': temp_cold_count / word_count,
            'humidity_dry_score': humidity_dry_count / word_count,
            'humidity_wet_score': humidity_wet_count / word_count,
            'wind_score': wind_count / word_count,
            'movement_score': movement_count / word_count,
            'sensory_score': sensory_count / word_count,
            'total_sensory_score': total_sensory / word_count,
            'dominant_temperature': 'hot' if temp_hot_count > temp_cold_count else 'cold' if temp_cold_count > 0 else 'neutral',
            'dominant_humidity': 'wet' if humidity_wet_count > humidity_dry_count else 'dry' if humidity_dry_count > 0 else 'neutral',
            'has_wind': wind_count > 0,
            'has_movement': movement_count > 0
        }
        
        return analysis


class SensorDataInferrer:
    """
    Infer realistic sensor data from literary passages.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def infer_sensor_data(self, passage: Dict) -> Dict:
        """
        Infer sensor readings from literary passage.
        
        Args:
            passage: Literary passage with sensory analysis
            
        Returns:
            dict: Inferred sensor data
        """
        analysis = passage['sensory_analysis']
        
        # Infer temperature
        temperature = self._infer_temperature(analysis)
        
        # Infer humidity  
        humidity = self._infer_humidity(analysis, temperature)
        
        # Infer wind direction
        wind_direction = self._infer_wind_direction(passage['text'], analysis)
        
        # Infer IMU data (movement)
        imu_data = self._infer_imu_data(analysis, passage['text'])
        
        # Extract contextual information
        context = self._extract_context(passage['text'])
        
        sensor_data = {
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'wind_direction': round(wind_direction, 3),
            'imu': [round(x, 3) for x in imu_data],
            'context': context,
            'confidence': self._calculate_confidence(analysis),
            'source_analysis': analysis
        }
        
        return sensor_data
    
    def _infer_temperature(self, analysis: Dict) -> float:
        """Infer temperature from sensory analysis."""
        base_temp = 20.0  # Default moderate temperature
        
        if analysis['dominant_temperature'] == 'hot':
            if analysis['temperature_hot_score'] > 0.02:
                base_temp = np.random.normal(30, 5)  # Hot day
            else:
                base_temp = np.random.normal(25, 3)  # Warm day
        elif analysis['dominant_temperature'] == 'cold':
            if analysis['temperature_cold_score'] > 0.02:
                base_temp = np.random.normal(5, 5)   # Cold day
            else:
                base_temp = np.random.normal(15, 3)  # Cool day
        else:
            base_temp = np.random.normal(20, 8)      # Variable
        
        return max(-10, min(40, base_temp))  # Reasonable bounds
    
    def _infer_humidity(self, analysis: Dict, temperature: float) -> float:
        """Infer humidity from sensory analysis and temperature."""
        base_humidity = 50.0
        
        if analysis['dominant_humidity'] == 'wet':
            if analysis['humidity_wet_score'] > 0.01:
                base_humidity = np.random.normal(80, 10)
            else:
                base_humidity = np.random.normal(65, 8)
        elif analysis['dominant_humidity'] == 'dry':
            if analysis['humidity_dry_score'] > 0.01:
                base_humidity = np.random.normal(30, 10)
            else:
                base_humidity = np.random.normal(45, 8)
        else:
            # Temperature-dependent default
            if temperature > 25:
                base_humidity = np.random.normal(60, 15)
            else:
                base_humidity = np.random.normal(55, 15)
        
        return max(10, min(100, base_humidity))
    
    def _infer_wind_direction(self, text: str, analysis: Dict) -> float:
        """Infer wind direction from text content."""
        # Default random direction
        base_direction = np.random.uniform(0, 2 * np.pi)
        
        # Look for directional keywords
        text_lower = text.lower()
        
        directional_hints = {
            'east': 0, 'eastern': 0, 'sunrise': 0,
            'south': np.pi/2, 'southern': np.pi/2,
            'west': np.pi, 'western': np.pi, 'sunset': np.pi,
            'north': 3*np.pi/2, 'northern': 3*np.pi/2,
            'front': 0, 'forward': 0, 'ahead': 0,
            'back': np.pi, 'behind': np.pi, 'rear': np.pi,
            'right': np.pi/2, 'rightward': np.pi/2,
            'left': 3*np.pi/2, 'leftward': 3*np.pi/2
        }
        
        for hint, direction in directional_hints.items():
            if hint in text_lower:
                return direction + np.random.normal(0, 0.3)  # Add some noise
        
        return base_direction
    
    def _infer_imu_data(self, analysis: Dict, text: str) -> List[float]:
        """Infer IMU data from movement analysis."""
        # Base stationary IMU
        acc_x, acc_y, acc_z = 0, 0, 9.8
        gyro_x, gyro_y, gyro_z = 0, 0, 0
        
        # Adjust based on movement intensity
        movement_intensity = analysis['movement_score'] * 10
        
        if 'run' in text or 'sprint' in text or 'dash' in text:
            # Running motion
            acc_x = np.random.normal(0, 2 + movement_intensity)
            acc_y = np.random.normal(0, 2 + movement_intensity)
            acc_z = 9.8 + np.random.normal(0, 1)
            gyro_x = np.random.normal(0, 0.5)
            gyro_y = np.random.normal(0, 0.5)
            gyro_z = np.random.normal(0, 0.2)
        elif 'walk' in text or 'stroll' in text:
            # Walking motion
            acc_x = np.random.normal(0, 1 + movement_intensity)
            acc_y = np.random.normal(0, 1 + movement_intensity)
            acc_z = 9.8 + np.random.normal(0, 0.5)
            gyro_x = np.random.normal(0, 0.2)
            gyro_y = np.random.normal(0, 0.2)
            gyro_z = np.random.normal(0, 0.1)
        else:
            # Minimal movement
            acc_x = np.random.normal(0, 0.5)
            acc_y = np.random.normal(0, 0.5)
            acc_z = 9.8 + np.random.normal(0, 0.2)
            gyro_x = np.random.normal(0, 0.1)
            gyro_y = np.random.normal(0, 0.1)
            gyro_z = np.random.normal(0, 0.05)
        
        return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    
    def _extract_context(self, text: str) -> Dict:
        """Extract contextual information from text."""
        text_lower = text.lower()
        
        # Time inference
        time_keywords = {
            'dawn': 'dawn', 'morning': 'morning', 'am': 'morning',
            'noon': 'noon', 'afternoon': 'afternoon', 'pm': 'afternoon',
            'evening': 'evening', 'night': 'night', 'midnight': 'midnight'
        }
        
        detected_time = 'afternoon'  # Default
        for keyword, time_label in time_keywords.items():
            if keyword in text_lower:
                detected_time = time_label
                break
        
        # Weather inference
        weather_keywords = {
            'rain': 'rain', 'snow': 'snow', 'clear': 'clear', 'cloudy': 'cloudy',
            'fog': 'fog', 'wind': 'windy', 'storm': 'storm'
        }
        
        detected_weather = 'clear'  # Default
        for keyword, weather_label in weather_keywords.items():
            if keyword in text_lower:
                detected_weather = weather_label
                break
        
        # Scenario inference
        scenario_keywords = {
            'forest': 'forest_exploration', 'mountain': 'mountain_climbing', 'ocean': 'beach_walking', 'beach': 'beach_walking',
            'city': 'city_walking', 'street': 'city_walking', 'park': 'park_stroll',
            'river': 'river_walk', 'field': 'field_crossing'
        }
        
        detected_scenario = 'city_walking'  # Default
        for keyword, scenario_label in scenario_keywords.items():
            if keyword in text_lower:
                detected_scenario = scenario_label
                break
        
        return {
            'scenario': detected_scenario,
            'time': detected_time,
            'weather': detected_weather
        }
    
    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence score for sensor inference."""
        total_score = analysis['total_sensory_score']
        
        # Higher sensory content = higher confidence
        if total_score > 0.05:
            return min(0.9, 0.5 + total_score * 5)
        else:
            return max(0.3, total_score * 10)


class CorpusBasedDatasetGenerator:
    """
    Main class for generating dataset from literary corpus.
    """
    
    def __init__(self):
        self.analyzer = CorpusAnalyzer()
        self.inferrer = SensorDataInferrer()
        self.logger = logging.getLogger(__name__)
    
    def generate_from_corpus(self, 
                           corpus_path: str,
                           output_path: str = None,
                           min_confidence: float = 0.4,
                           max_examples: int = None) -> List[Dict]:
        """
        Generate complete dataset from literary corpus.
        
        Args:
            corpus_path: Path to corpus file or directory
            output_path: Optional output file path
            min_confidence: Minimum confidence threshold
            max_examples: Maximum number of examples
            
        Returns:
            list: Generated dataset
        """
        self.logger.info(f"Processing corpus from {corpus_path}")
        
        # Load and analyze corpus
        texts = self.analyzer.load_corpus(corpus_path)
        sensory_passages = self.analyzer.extract_sensory_passages(texts)
        
        # Generate dataset
        dataset = []
        for passage in sensory_passages:
            # Infer sensor data
            sensor_data = self.inferrer.infer_sensor_data(passage)
            
            # Filter by confidence
            if sensor_data['confidence'] < min_confidence:
                continue
            
            # Create dataset entry
            entry = {
                'id': passage['id'],
                'sensor_data': {
                    'temperature': sensor_data['temperature'],
                    'humidity': sensor_data['humidity'], 
                    'wind_direction': sensor_data['wind_direction'],
                    'imu': sensor_data['imu'],
                    'context': sensor_data['context']
                },
                'target_paragraph': passage['text'],
                'metadata': {
                    'source': 'corpus',
                    'confidence': sensor_data['confidence'],
                    'sensory_analysis': sensor_data['source_analysis'],
                    'original_length': passage['length'],
                    'generated_at': pd.Timestamp.now().isoformat()
                }
            }
            
            dataset.append(entry)
            
            # Limit dataset size
            if max_examples and len(dataset) >= max_examples:
                break
        
        self.logger.info(f"Generated {len(dataset)} examples from corpus")
        
        # Save if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Dataset saved to {output_path}")
        
        return dataset
    
    def analyze_corpus_statistics(self, corpus_path: str) -> Dict:
        """Generate statistics about the corpus."""
        texts = self.analyzer.load_corpus(corpus_path)
        sensory_passages = self.analyzer.extract_sensory_passages(texts)
        
        # Calculate statistics
        total_passages = len(sensory_passages)
        avg_length = np.mean([p['length'] for p in sensory_passages])
        
        # Sensory content distribution
        temp_hot = sum(1 for p in sensory_passages if p['sensory_analysis']['temperature_hot_score'] > 0.01)
        temp_cold = sum(1 for p in sensory_passages if p['sensory_analysis']['temperature_cold_score'] > 0.01)
        has_wind = sum(1 for p in sensory_passages if p['sensory_analysis']['has_wind'])
        has_movement = sum(1 for p in sensory_passages if p['sensory_analysis']['has_movement'])
        
        return {
            'total_original_texts': len(texts),
            'sensory_passages': total_passages,
            'average_passage_length': avg_length,
            'temperature_hot_passages': temp_hot,
            'temperature_cold_passages': temp_cold,
            'wind_passages': has_wind,
            'movement_passages': has_movement,
            'high_sensory_passages': sum(1 for p in sensory_passages if p['sensory_analysis']['total_sensory_score'] > 0.05)
        }


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    generator = CorpusBasedDatasetGenerator()
    
    # Example: process corpus directory
    corpus_path = "path/to/your/literary/corpus"  # Change this
    output_path = "data/corpus_based_dataset.json"
    
    # Generate statistics first
    print("Analyzing corpus...")
    stats = generator.analyze_corpus_statistics(corpus_path)
    print(f"Corpus Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generator.generate_from_corpus(
        corpus_path=corpus_path,
        output_path=output_path,
        min_confidence=0.4,
        max_examples=1000
    )
    
    print(f"\nGenerated {len(dataset)} examples")
    
    # Show example
    if dataset:
        example = dataset[0]
        print(f"\nExample generated data:")
        print(f"Temperature: {example['sensor_data']['temperature']}°C")
        print(f"Humidity: {example['sensor_data']['humidity']}%")
        print(f"Confidence: {example['metadata']['confidence']:.2f}")
        print(f"Text: {example['target_paragraph'][:200]}...")


if __name__ == "__main__":
    main()