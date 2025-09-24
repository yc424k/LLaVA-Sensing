#!/usr/bin/env python3
"""
Simple Novel corpus processor without heavy dependencies.
"""

import os
import json
import re
import random
import math
from pathlib import Path
from collections import defaultdict


class SimpleNovelProcessor:
    """Simple processor for Novel corpus."""
    
    def __init__(self, novel_dir="/home/yc424k/LLaVA-NeXT/Novel"):
        self.novel_dir = Path(novel_dir)
        self.modernist_dir = self.novel_dir / "Modernist_Novel"
        self.travel_dir = self.novel_dir / "Travel_Novel"
        
        # Sensory keywords for analysis
        self.temp_keywords = {
            'hot': ['hot', 'warm', 'burning', 'blazing', 'scorching', 'sweltering', 'heated', 'sultry'],
            'cold': ['cold', 'cool', 'chilly', 'freezing', 'frozen', 'icy', 'frigid', 'bitter']
        }
        
        self.humidity_keywords = {
            'humid': ['humid', 'moist', 'damp', 'wet', 'soggy', 'steamy', 'muggy'],
            'dry': ['dry', 'arid', 'parched', 'dusty', 'drought']
        }
        
        self.wind_keywords = ['wind', 'breeze', 'gust', 'blow', 'blowing', 'draft', 'air', 'current']
        self.movement_keywords = ['walk', 'run', 'move', 'step', 'stride', 'pace', 'march', 'wander']
        
    def extract_metadata(self, file_path):
        """Extract metadata from novel file."""
        metadata = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines[:15]:
                line = line.strip()
                if ':' in line and not line.startswith('***'):
                    key, value = line.split(':', 1)
                    key_clean = key.strip().lower().replace(' ', '_')
                    # Skip source_link and source link related fields
                    if 'source' in key_clean and 'link' in key_clean:
                        continue
                    metadata[key_clean] = value.strip()
                elif line.startswith('*** START OF'):
                    break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return metadata
    
    def extract_content(self, file_path):
        """Extract main content from novel."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            start_marker = "*** START OF THIS TEXT"
            start_idx = content.find(start_marker)
            
            if start_idx != -1:
                start_idx = content.find('\n', start_idx)
                content = content[start_idx:].strip()
            
            return content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def split_into_passages(self, content, min_length=150, max_length=500):
        """Split content into passages."""
        passages = []
        paragraphs = content.split('\n\n')
        current_passage = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_passage) + len(paragraph) > max_length:
                if len(current_passage) >= min_length:
                    passages.append(current_passage.strip())
                current_passage = paragraph
            else:
                if current_passage:
                    current_passage += "\n\n" + paragraph
                else:
                    current_passage = paragraph
        
        if len(current_passage) >= min_length:
            passages.append(current_passage.strip())
        
        return passages
    
    def analyze_sensory_content(self, text):
        """Analyze sensory content in text."""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return {'sensory_score': 0, 'details': {}}
        
        # Count keywords
        temp_hot = sum(1 for word in self.temp_keywords['hot'] if word in text_lower)
        temp_cold = sum(1 for word in self.temp_keywords['cold'] if word in text_lower)
        humidity_wet = sum(1 for word in self.humidity_keywords['humid'] if word in text_lower)
        humidity_dry = sum(1 for word in self.humidity_keywords['dry'] if word in text_lower)
        wind_count = sum(1 for word in self.wind_keywords if word in text_lower)
        movement_count = sum(1 for word in self.movement_keywords if word in text_lower)
        
        total_sensory = temp_hot + temp_cold + humidity_wet + humidity_dry + wind_count + movement_count
        sensory_score = total_sensory / word_count if word_count > 0 else 0
        
        return {
            'sensory_score': sensory_score,
            'details': {
                'temperature_hot': temp_hot,
                'temperature_cold': temp_cold,
                'humidity_wet': humidity_wet,
                'humidity_dry': humidity_dry,
                'wind': wind_count,
                'movement': movement_count,
                'total_words': word_count
            }
        }
    
    def infer_sensor_data(self, text, analysis):
        """Infer sensor data from text analysis."""
        details = analysis['details']
        
        # Temperature inference
        if details['temperature_hot'] > details['temperature_cold']:
            if details['temperature_hot'] > 2:
                temperature = random.uniform(25, 35)
            else:
                temperature = random.uniform(20, 28)
        elif details['temperature_cold'] > 0:
            if details['temperature_cold'] > 2:
                temperature = random.uniform(-5, 10)
            else:
                temperature = random.uniform(5, 18)
        else:
            temperature = random.uniform(10, 25)
        
        # Humidity inference
        if details['humidity_wet'] > details['humidity_dry']:
            humidity = random.uniform(60, 90)
        elif details['humidity_dry'] > 0:
            humidity = random.uniform(20, 45)
        else:
            humidity = random.uniform(40, 70)
        
        # Wind direction (random)
        wind_direction = random.uniform(0, 2 * math.pi)
        
        # IMU data based on movement
        if details['movement'] > 2:
            # High movement
            acc_std = 2.0
            gyro_std = 0.3
        elif details['movement'] > 0:
            # Some movement
            acc_std = 1.0
            gyro_std = 0.15
        else:
            # Little movement
            acc_std = 0.3
            gyro_std = 0.05
        
        imu_data = [
            random.gauss(0, acc_std),      # acc_x
            random.gauss(0, acc_std),      # acc_y
            9.8 + random.gauss(0, acc_std * 0.2),  # acc_z
            random.gauss(0, gyro_std),     # gyro_x
            random.gauss(0, gyro_std),     # gyro_y
            random.gauss(0, gyro_std * 0.5) # gyro_z
        ]
        
        # Context inference
        text_lower = text.lower()
        
        # Time
        time_keywords = {
            'dawn': 'dawn', 'morning': 'morning', 'noon': 'noon',
            'afternoon': 'afternoon', 'evening': 'evening', 'night': 'night'
        }
        detected_time = 'afternoon'
        for keyword, time_label in time_keywords.items():
            if keyword in text_lower:
                detected_time = time_label
                break
        
        # Weather
        if any(word in text_lower for word in ['rain', 'raining']):
            weather = 'rain'
        elif any(word in text_lower for word in ['snow', 'snowing']):
            weather = 'snow'
        elif any(word in text_lower for word in ['cloud', 'cloudy', 'overcast']):
            weather = 'cloudy'
        else:
            weather = 'clear'
        
        # Scenario
        if any(word in text_lower for word in ['forest', 'tree', 'wood']):
            scenario = 'forest_exploration'
        elif any(word in text_lower for word in ['mountain', 'hill', 'climb']):
            scenario = 'mountain_climbing'
        elif any(word in text_lower for word in ['sea', 'ocean', 'beach', 'shore']):
            scenario = 'beach_walking'
        elif any(word in text_lower for word in ['city', 'street', 'building']):
            scenario = 'city_walking'
        else:
            scenario = 'general_walking'
        
        return {
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'wind_direction': round(wind_direction, 3),
            'imu': [round(x, 3) for x in imu_data],
            'context': {
                'scenario': scenario,
                'time': detected_time,
                'weather': weather
            }
        }
    
    def process_novel_file(self, file_path, style_category):
        """Process single novel file."""
        print(f"Processing {file_path.name}...")
        
        metadata = self.extract_metadata(file_path)
        content = self.extract_content(file_path)
        
        if not content:
            return []
        
        passages = self.split_into_passages(content)
        examples = []
        
        for i, passage in enumerate(passages):
            analysis = self.analyze_sensory_content(passage)
            
            # Skip passages with very low sensory content
            if analysis['sensory_score'] < 0.01:
                continue
            
            sensor_data = self.infer_sensor_data(passage, analysis)
            
            example = {
                'id': f"{file_path.stem}_passage_{i:03d}",
                'sensor_data': sensor_data,
                'target_paragraph': passage,
                'metadata': {
                    'source_file': file_path.name,
                    'style_category': style_category,
                    'sensory_analysis': analysis,
                    'novel_metadata': metadata,
                    'passage_index': i
                }
            }
            
            examples.append(example)
        
        print(f"Generated {len(examples)} examples from {file_path.name}")
        return examples
    
    def process_corpus(self, max_files_per_category=3):
        """Process entire corpus."""
        all_examples = []
        
        # Process Modernist novels
        if self.modernist_dir.exists():
            modernist_files = list(self.modernist_dir.glob("*.txt"))[:max_files_per_category]
            print(f"Processing {len(modernist_files)} modernist novels...")
            
            for file_path in modernist_files:
                examples = self.process_novel_file(file_path, "modernist")
                all_examples.extend(examples)
        
        # Process Travel novels
        if self.travel_dir.exists():
            travel_files = list(self.travel_dir.glob("*.txt"))[:max_files_per_category]
            print(f"Processing {len(travel_files)} travel novels...")
            
            for file_path in travel_files:
                examples = self.process_novel_file(file_path, "travel")
                all_examples.extend(examples)
        
        return all_examples
    
    def save_dataset(self, examples, output_file="data/novel_dataset.json"):
        """Save dataset to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Shuffle examples
        random.shuffle(examples)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(examples)} examples to {output_file}")
        
        # Generate simple statistics
        stats = self.generate_stats(examples)
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def generate_stats(self, examples):
        """Generate simple statistics."""
        if not examples:
            return {}
        
        style_counts = defaultdict(int)
        temp_values = []
        humidity_values = []
        passage_lengths = []
        
        for example in examples:
            style_counts[example['metadata']['style_category']] += 1
            temp_values.append(example['sensor_data']['temperature'])
            humidity_values.append(example['sensor_data']['humidity'])
            passage_lengths.append(len(example['target_paragraph']))
        
        return {
            'total_examples': len(examples),
            'style_distribution': dict(style_counts),
            'temperature_range': [min(temp_values), max(temp_values)],
            'humidity_range': [min(humidity_values), max(humidity_values)],
            'avg_passage_length': sum(passage_lengths) // len(passage_lengths)
        }


def main():
    """Main function."""
    processor = SimpleNovelProcessor()
    
    print("Processing Novel corpus...")
    examples = processor.process_corpus(max_files_per_category=3)
    
    print("Saving dataset...")
    stats = processor.save_dataset(examples)
    
    print("\n=== Dataset Summary ===")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Style distribution: {stats['style_distribution']}")
    print(f"Temperature range: {stats['temperature_range']}")
    print(f"Humidity range: {stats['humidity_range']}")
    print(f"Average passage length: {stats['avg_passage_length']} characters")
    
    # Show sample
    if examples:
        sample = examples[0]
        print(f"\n=== Sample Example ===")
        print(f"Source: {sample['metadata']['source_file']}")
        print(f"Style: {sample['metadata']['style_category']}")
        print(f"Temperature: {sample['sensor_data']['temperature']}°C")
        print(f"Humidity: {sample['sensor_data']['humidity']}%")
        print(f"Context: {sample['sensor_data']['context']}")
        print(f"Text preview: {sample['target_paragraph'][:200]}...")


if __name__ == "__main__":
    main()