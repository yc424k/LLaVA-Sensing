#!/usr/bin/env python3
"""
Hybrid Novel corpus processor that preserves original text while using LLM for environmental analysis.
Combines the best of both approaches: original literary text + sophisticated environmental inference.
"""

import os
import json
import re
import random
import math
import requests
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime

# Simple progress bar implementation
class ProgressBar:
    def __init__(self, total, description="Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, count=1):
        self.current += count
        self.show()
        
    def show(self):
        if self.total == 0:
            return
            
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
        else:
            eta_str = "--:--"
            
        # Progress bar
        bar_length = 30
        filled = int(bar_length * self.current / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r{self.description}: {bar} {percent:5.1f}% ({self.current}/{self.total}) ETA: {eta_str}", end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


class HybridNovelProcessor:
    """
    Hybrid processor that:
    1. Uses original novel text (preserves literary style and author's voice)
    2. Uses LLM for sophisticated environmental analysis (temperature, weather, context)
    3. Generates realistic sensor data based on LLM analysis
    """
    
    def __init__(self, novel_dir="/home/yc424k/LLaVA-Sensing/Novel", 
                 use_ollama=True, ollama_model="llama3.2:3b"):
        self.novel_dir = Path(novel_dir)
        self.modernist_dir = self.novel_dir / "Modernist_Novel"
        self.travel_dir = self.novel_dir / "Travel_Novel"
        
        # LLM configuration
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Fallback keywords for when LLM fails
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

    def _call_ollama(self, prompt: str, temperature: float = 0.3) -> str:
        """Call Ollama API for environmental analysis."""
        try:
            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 300
                }
            }
            
            response = requests.post(self.ollama_url, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Ollama API call failed: {e}")
            return ""

    def analyze_environment_with_llm(self, text: str) -> Dict:
        """
        Use LLM to analyze environmental information in literary text.
        This is the key improvement over simple keyword matching.
        """
        prompt = f"""Analyze this literary text and extract environmental information that could be measured by sensors.
Focus on subtle environmental cues and atmospheric details.

Text to analyze:
"{text}"

Please respond with ONLY a valid JSON object containing:
{{
  "temperature_celsius": <number between -20 to 40>,
  "humidity_percent": <number between 0 to 100>,
  "weather_condition": "clear/cloudy/rain/snow/fog/windy/storm",
  "time_of_day": "dawn/morning/noon/afternoon/evening/night",
  "location_type": "city/forest/beach/mountain/indoor/field/river",
  "movement_intensity": "still/light/moderate/active",
  "atmospheric_details": {{
    "wind_present": true/false,
    "temperature_feel": "cold/cool/mild/warm/hot",
    "humidity_feel": "dry/moderate/humid/very_humid",
    "overall_mood": "calm/energetic/tense/peaceful"
  }},
  "confidence": <0.0 to 1.0>
}}

Extract information only if clearly indicated in the text. If uncertain, use moderate values and lower confidence."""

        try:
            if self.use_ollama:
                response = self._call_ollama(
                    "You are an expert at analyzing literary text for environmental details. Always respond with valid JSON only.\n\n" + prompt,
                    temperature=0.3
                )
            else:
                # Could add OpenAI support here if needed
                response = ""
            
            if response:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        return result
                    except json.JSONDecodeError:
                        pass
            
            # Fallback to simple analysis if LLM fails
            print("LLM analysis failed, falling back to simple analysis")
            return self.simple_environmental_analysis(text)
            
        except Exception as e:
            print(f"Environmental analysis failed: {e}")
            return self.simple_environmental_analysis(text)

    def simple_environmental_analysis(self, text: str) -> Dict:
        """Fallback environmental analysis using keyword patterns."""
        text_lower = text.lower()
        
        # Temperature analysis
        hot_count = sum(1 for word in self.temp_keywords['hot'] if word in text_lower)
        cold_count = sum(1 for word in self.temp_keywords['cold'] if word in text_lower)
        
        if hot_count > cold_count:
            temp_celsius = random.uniform(20, 30)
            temp_feel = "warm" if hot_count == 1 else "hot"
        elif cold_count > 0:
            temp_celsius = random.uniform(0, 15)
            temp_feel = "cool" if cold_count == 1 else "cold"
        else:
            temp_celsius = random.uniform(10, 20)
            temp_feel = "mild"
        
        # Humidity analysis
        humid_count = sum(1 for word in self.humidity_keywords['humid'] if word in text_lower)
        dry_count = sum(1 for word in self.humidity_keywords['dry'] if word in text_lower)
        
        if humid_count > dry_count:
            humidity_percent = random.uniform(65, 85)
            humidity_feel = "humid"
        elif dry_count > 0:
            humidity_percent = random.uniform(25, 45)
            humidity_feel = "dry"
        else:
            humidity_percent = random.uniform(45, 65)
            humidity_feel = "moderate"
        
        # Weather analysis
        weather_condition = "clear"
        if any(word in text_lower for word in ['rain', 'raining']):
            weather_condition = "rain"
        elif any(word in text_lower for word in ['snow', 'snowing']):
            weather_condition = "snow"
        elif any(word in text_lower for word in ['fog', 'mist']):
            weather_condition = "fog"
        elif any(word in text_lower for word in self.wind_keywords):
            weather_condition = "windy"
        elif any(word in text_lower for word in ['cloud', 'cloudy']):
            weather_condition = "cloudy"
        
        # Time analysis
        time_of_day = "afternoon"
        if any(word in text_lower for word in ['morning', 'dawn']):
            time_of_day = "morning"
        elif any(word in text_lower for word in ['noon', 'midday']):
            time_of_day = "noon"
        elif any(word in text_lower for word in ['evening', 'dusk']):
            time_of_day = "evening"
        elif any(word in text_lower for word in ['night', 'midnight']):
            time_of_day = "night"
        
        # Location analysis
        location_type = "city"
        if any(word in text_lower for word in ['forest', 'woods', 'trees']):
            location_type = "forest"
        elif any(word in text_lower for word in ['beach', 'ocean', 'sea']):
            location_type = "beach"
        elif any(word in text_lower for word in ['mountain', 'hill']):
            location_type = "mountain"
        elif any(word in text_lower for word in ['river', 'stream']):
            location_type = "river"
        elif any(word in text_lower for word in ['field', 'meadow']):
            location_type = "field"
        
        # Movement analysis
        movement_count = sum(1 for word in self.movement_keywords if word in text_lower)
        if movement_count > 3:
            movement_intensity = "active"
        elif movement_count > 1:
            movement_intensity = "moderate"
        elif movement_count > 0:
            movement_intensity = "light"
        else:
            movement_intensity = "still"
        
        return {
            "temperature_celsius": round(temp_celsius, 1),
            "humidity_percent": round(humidity_percent, 1),
            "weather_condition": weather_condition,
            "time_of_day": time_of_day,
            "location_type": location_type,
            "movement_intensity": movement_intensity,
            "atmospheric_details": {
                "wind_present": any(word in text_lower for word in self.wind_keywords),
                "temperature_feel": temp_feel,
                "humidity_feel": humidity_feel,
                "overall_mood": "calm"
            },
            "confidence": 0.6  # Lower confidence for fallback method
        }

    def generate_sensor_data_from_analysis(self, env_analysis: Dict) -> Dict:
        """Generate realistic sensor data based on environmental analysis."""
        
        # Use LLM analysis results with safe defaults
        temperature = env_analysis.get("temperature_celsius", 15.0)
        humidity = env_analysis.get("humidity_percent", 50.0)
        
        # Ensure temperature is a valid number
        if temperature is None:
            temperature = 15.0
        elif isinstance(temperature, list):
            temperature = temperature[0] if temperature and isinstance(temperature[0], (int, float)) else 15.0
        elif not isinstance(temperature, (int, float)):
            try:
                temperature = float(temperature)
            except (ValueError, TypeError):
                temperature = 15.0
        
        # Ensure humidity is a valid number
        if humidity is None:
            humidity = 50.0
        elif isinstance(humidity, list):
            humidity = humidity[0] if humidity and isinstance(humidity[0], (int, float)) else 50.0
        elif not isinstance(humidity, (int, float)):
            try:
                humidity = float(humidity)
            except (ValueError, TypeError):
                humidity = 50.0
            
        # Convert to float and ensure valid ranges
        temperature = max(-20.0, min(40.0, float(temperature)))
        humidity = max(0.0, min(100.0, float(humidity)))
        
        # Wind direction (random but influenced by location)
        location = env_analysis.get("location_type", "city")
        
        # Ensure location is a string (fix for unhashable type error)
        if isinstance(location, list):
            location = location[0] if location else "city"
        elif not isinstance(location, str):
            location = "city"
            
        if location == "beach":
            wind_direction = random.uniform(0, math.pi/2)  # Ocean winds
        elif location == "mountain":
            wind_direction = random.uniform(0, 2*math.pi)  # Variable mountain winds
        else:
            wind_direction = random.uniform(0, 2*math.pi)
        
        # IMU data based on movement intensity
        movement = env_analysis.get("movement_intensity", "light")
        
        # Ensure movement is a string (fix for potential type errors)
        if isinstance(movement, list):
            movement = movement[0] if movement else "light"
        elif not isinstance(movement, str):
            movement = "light"
            
        if movement == "active":
            acc_std, gyro_std = 2.0, 0.3
        elif movement == "moderate":
            acc_std, gyro_std = 1.0, 0.15
        elif movement == "light":
            acc_std, gyro_std = 0.5, 0.1
        else:  # still
            acc_std, gyro_std = 0.2, 0.05
        
        imu_data = [
            random.gauss(0, acc_std),                    # acc_x
            random.gauss(0, acc_std),                    # acc_y  
            9.8 + random.gauss(0, acc_std * 0.2),       # acc_z
            random.gauss(0, gyro_std),                   # gyro_x
            random.gauss(0, gyro_std),                   # gyro_y
            random.gauss(0, gyro_std * 0.5)              # gyro_z
        ]
        
        # Create context mapping
        scenario_mapping = {
            "city": "city_walking",
            "forest": "forest_exploration", 
            "beach": "beach_walking",
            "mountain": "mountain_climbing",
            "river": "riverside_walking",
            "field": "field_crossing"
        }
        
        scenario = scenario_mapping.get(location, "general_walking")
        
        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "wind_direction": round(wind_direction, 3),
            "imu": [round(x, 3) for x in imu_data],
            "context": {
                "scenario": scenario,
                "time": env_analysis.get("time_of_day", "afternoon"),
                "weather": env_analysis.get("weather_condition", "clear")
            }
        }

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
                    if 'source' not in key_clean or 'link' not in key_clean:
                        metadata[key_clean] = value.strip()
                elif line.startswith('*** START OF'):
                    break
        except Exception as e:
            print(f"Error reading metadata from {file_path}: {e}")
            
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
            print(f"Error reading content from {file_path}: {e}")
            return ""

    def split_into_passages(self, content, min_length=500, max_length=700):
        """Split content into meaningful passages."""
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

    def process_novel_file(self, file_path, style_category, max_examples_per_novel=100):
        """Process single novel file using hybrid approach."""
        print(f"\n📚 Processing {file_path.name} with hybrid approach...")
        print(f"🎯 Max examples per novel: {max_examples_per_novel}")
        
        metadata = self.extract_metadata(file_path)
        content = self.extract_content(file_path)
        
        if not content:
            print(f"❌ No content found in {file_path.name}")
            return []
        
        passages = self.split_into_passages(content)
        print(f"📄 Split into {len(passages)} passages")
        
        if len(passages) == 0:
            print(f"❌ No valid passages found in {file_path.name}")
            return []
        
        # Initialize progress bar for this file
        progress = ProgressBar(len(passages), f"Processing {file_path.name}")
        examples = []
        skipped_count = 0
        
        for i, passage in enumerate(passages):
            # Update progress
            progress.update()
            
            # Check if we've reached the limit for this novel
            if len(examples) >= max_examples_per_novel:
                print(f"\n🚫 Reached limit of {max_examples_per_novel} examples for {file_path.name}")
                print(f"📊 Processed {i+1}/{len(passages)} passages, stopping early")
                break
            
            # KEY: Use LLM for environmental analysis
            env_analysis = self.analyze_environment_with_llm(passage)
            
            # Skip passages with very low confidence
            confidence = env_analysis.get("confidence", 0)
            if isinstance(confidence, (int, float)) and confidence < 0.3:
                skipped_count += 1
                continue
            
            # Generate sensor data from LLM analysis
            sensor_data = self.generate_sensor_data_from_analysis(env_analysis)
            
            # KEY: Keep original passage text (preserve literary style)
            example = {
                'id': f"{file_path.stem}_hybrid_{i:03d}",
                'sensor_data': sensor_data,
                'target_paragraph': passage,  # Original text preserved!
                'metadata': {
                    'source_file': file_path.name,
                    'style_category': style_category,
                    'environmental_analysis': env_analysis,  # LLM analysis results
                    'novel_metadata': metadata,
                    'passage_index': i,
                    'generation_method': 'hybrid',
                    'llm_confidence': env_analysis.get("confidence", 0)
                }
            }
            
            examples.append(example)
        
        print(f"✅ Generated {len(examples)} examples, skipped {skipped_count} low-confidence passages")
        return examples

    def process_corpus(self, max_files_per_category=10, max_examples_per_novel=100):
        """Process entire corpus with hybrid approach."""
        print(f"\n🚀 Starting Hybrid Novel Processing")
        print(f"📁 Novel directory: {self.novel_dir}")
        print(f"📊 Max files per category: {max_files_per_category}")
        print(f"🎯 Max examples per novel: {max_examples_per_novel}")
        print(f"🤖 Using Ollama model: {self.ollama_model}")
        
        all_examples = []
        all_files = []
        
        # Collect all files first for overall progress
        if self.modernist_dir.exists():
            modernist_files = list(self.modernist_dir.glob("*.txt"))[:max_files_per_category]
            all_files.extend([(f, "modernist") for f in modernist_files])
        
        if self.travel_dir.exists():
            travel_files = list(self.travel_dir.glob("*.txt"))[:max_files_per_category]
            all_files.extend([(f, "travel") for f in travel_files])
        
        if not all_files:
            print(f"❌ No novel files found in {self.novel_dir}")
            return []
        
        print(f"📚 Found {len(all_files)} total files to process")
        
        # Calculate estimated max examples
        estimated_max = len(all_files) * max_examples_per_novel
        print(f"📈 Estimated max examples: {estimated_max}")
        
        # Overall progress tracking
        overall_progress = ProgressBar(len(all_files), "Overall Progress")
        
        for file_path, style_category in all_files:
            examples = self.process_novel_file(file_path, style_category, max_examples_per_novel)
            all_examples.extend(examples)
            overall_progress.update()
            
            # Show current totals
            print(f"📈 Running total: {len(all_examples)} examples generated")
        
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Total examples generated: {len(all_examples)}")
        print(f"📁 From {len(all_files)} novel files")
        print(f"📊 Average per novel: {len(all_examples) // len(all_files):.1f}")
        
        return all_examples

    def save_dataset(
        self,
        examples,
        output_file="stage0_data_processing/data_generation/data/processed/hybrid_novel_dataset.json",
        chunk_size=100,
    ):
        """Save dataset to file with chunking for better management."""
        base_dir = os.path.dirname(output_file)
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        
        # Create base directory and chunk subdirectory
        chunk_dir = os.path.join(base_dir, f"{base_name}_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Shuffle examples
        random.shuffle(examples)
        
        # Save in chunks
        total_chunks = (len(examples) + chunk_size - 1) // chunk_size
        chunk_files = []
        
        print(f"\n💾 Saving {len(examples)} examples in {total_chunks} chunks of {chunk_size} each...")
        
        # Progress bar for saving
        save_progress = ProgressBar(total_chunks, "Saving chunks")
        
        for i in range(0, len(examples), chunk_size):
            chunk_examples = examples[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            
            # Create chunk filename
            chunk_filename = f"{base_name}_chunk_{chunk_num:03d}.json"
            chunk_path = os.path.join(chunk_dir, chunk_filename)
            
            # Save chunk
            with open(chunk_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_examples, f, ensure_ascii=False, indent=2)
            
            chunk_files.append(chunk_filename)
            save_progress.update()
        
        # Create index file with chunk information
        index_data = {
            "total_examples": len(examples),
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_directory": f"{base_name}_chunks",
            "chunk_files": chunk_files,
            "created_at": datetime.now().isoformat(),
            "generation_method": "hybrid"
        }
        
        index_file = os.path.join(base_dir, f"{base_name}_index.json")
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        # Generate detailed statistics
        stats = self.generate_stats(examples)
        stats_file = os.path.join(base_dir, f"{base_name}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Summary
        print(f"\n✅ Dataset Saved Successfully!")
        print(f"📁 Chunk directory: {chunk_dir}")
        print(f"📄 Index file: {index_file}")
        print(f"📊 Stats file: {stats_file}")
        print(f"🔢 Total examples: {len(examples)}")
        print(f"📦 Total chunks: {total_chunks}")
        
        return stats
    
    def load_dataset_chunks(self, index_file_path):
        """Load dataset from chunked files using index."""
        with open(index_file_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        base_dir = os.path.dirname(index_file_path)
        chunk_dir = os.path.join(base_dir, index_data["chunk_directory"])
        
        all_examples = []
        chunk_files = index_data["chunk_files"]
        
        print(f"📦 Loading {index_data['total_examples']} examples from {len(chunk_files)} chunks...")
        
        load_progress = ProgressBar(len(chunk_files), "Loading chunks")
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_examples = json.load(f)
                    all_examples.extend(chunk_examples)
                load_progress.update()
            except Exception as e:
                print(f"❌ Error loading {chunk_file}: {e}")
        
        print(f"✅ Successfully loaded {len(all_examples)} examples total")
        return all_examples

    def generate_stats(self, examples):
        """Generate comprehensive statistics for hybrid dataset."""
        if not examples:
            return {}
        
        style_counts = defaultdict(int)
        confidence_scores = []
        temp_values = []
        humidity_values = []
        passage_lengths = []
        weather_counts = defaultdict(int)
        location_counts = defaultdict(int)
        
        for example in examples:
            metadata = example['metadata']
            sensor_data = example['sensor_data']
            
            style_counts[metadata['style_category']] += 1
            confidence_scores.append(metadata.get('llm_confidence', 0))
            temp_values.append(sensor_data['temperature'])
            humidity_values.append(sensor_data['humidity'])
            passage_lengths.append(len(example['target_paragraph']))
            weather_value = sensor_data['context']['weather']
            if isinstance(weather_value, str):
                weather_counts[weather_value] += 1
            else:
                weather_counts['unknown'] += 1
            location_counts[sensor_data['context']['scenario']] += 1
        
        return {
            'total_examples': len(examples),
            'generation_method': 'hybrid',
            'style_distribution': dict(style_counts),
            'average_llm_confidence': sum(confidence_scores) / len(confidence_scores),
            'temperature_range': [min(temp_values), max(temp_values)],
            'humidity_range': [min(humidity_values), max(humidity_values)],
            'avg_passage_length': sum(passage_lengths) // len(passage_lengths),
            'weather_distribution': dict(weather_counts),
            'location_distribution': dict(location_counts),
            'high_confidence_examples': sum(1 for conf in confidence_scores if conf > 0.7)
        }


def main():
    """Main function to demonstrate hybrid approach."""
    processor = HybridNovelProcessor()
    
    print("=== Hybrid Novel Processing ===")
    print("✅ Preserves original literary text")
    print("✅ Uses LLM for sophisticated environmental analysis")
    print("✅ Generates realistic sensor data")
    print()
    
    examples = processor.process_corpus(max_files_per_category=200, max_examples_per_novel=120)
    
    if examples:
        stats = processor.save_dataset(examples)
        
        print("\n=== Hybrid Dataset Summary ===")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Average LLM confidence: {stats['average_llm_confidence']:.2f}")
        print(f"Style distribution: {stats['style_distribution']}")
        print(f"High confidence examples: {stats['high_confidence_examples']}")
        print(f"Weather variety: {len(stats['weather_distribution'])} types")
        print(f"Location variety: {len(stats['location_distribution'])} types")
        
        # Show sample example
        sample = examples[0]
        print(f"\n=== Sample Hybrid Example ===")
        print(f"Source: {sample['metadata']['source_file']}")
        print(f"Style: {sample['metadata']['style_category']}")
        print(f"LLM Confidence: {sample['metadata']['llm_confidence']:.2f}")
        print(f"Temperature: {sample['sensor_data']['temperature']}°C")
        print(f"Context: {sample['sensor_data']['context']}")
        print(f"Original Text (preserved): {sample['target_paragraph'][:150]}...")
    else:
        print("No examples generated. Check novel directory path and Ollama setup.")


if __name__ == "__main__":
    main()
