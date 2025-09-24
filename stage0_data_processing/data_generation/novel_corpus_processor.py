"""
Process Novel corpus (Modernist_Novel + Travel_Novel) to generate training data
for sensor-to-literature model.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import pandas as pd

from corpus_based_generator import CorpusAnalyzer, SensorDataInferrer
from corpus_processor import LiteraryStyleClassifier, SensorContextExtractor, AdvancedSensorMapper


class NovelCorpusProcessor:
    """
    Specialized processor for the Novel corpus structure.
    """
    
    def __init__(self, novel_dir: str = "/home/yc424k/LLaVA-NeXT/Novel"):
        self.novel_dir = Path(novel_dir)
        self.modernist_dir = self.novel_dir / "Modernist_Novel"
        self.travel_dir = self.novel_dir / "Travel_Novel"
        
        # Initialize processing components
        self.analyzer = CorpusAnalyzer()
        self.inferrer = SensorDataInferrer()
        self.style_classifier = LiteraryStyleClassifier()
        self.context_extractor = SensorContextExtractor()
        self.sensor_mapper = AdvancedSensorMapper()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_novel_metadata(self, file_path: Path) -> Dict:
        """
        Extract metadata from novel file header.
        
        Args:
            file_path: Path to novel text file
            
        Returns:
            dict: Extracted metadata
        """
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Extract metadata from header
            for line in lines[:15]:  # Check first 15 lines
                line = line.strip()
                if ':' in line and not line.startswith('***'):
                    key, value = line.split(':', 1)
                    metadata[key.strip().lower().replace(' ', '_')] = value.strip()
                elif line.startswith('*** START OF'):
                    break
                    
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            
        return metadata
    
    def extract_novel_content(self, file_path: Path) -> str:
        """
        Extract main content from novel file, removing metadata header.
        
        Args:
            file_path: Path to novel text file
            
        Returns:
            str: Clean novel content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find start marker
            start_marker = "*** START OF THIS TEXT"
            start_idx = content.find(start_marker)
            
            if start_idx != -1:
                # Find end of start marker line
                start_idx = content.find('\n', start_idx)
                content = content[start_idx:].strip()
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    def split_into_passages(self, content: str, min_length: int = 150, max_length: int = 500) -> List[str]:
        """
        Split novel content into manageable passages.
        
        Args:
            content: Novel text content
            min_length: Minimum passage length
            max_length: Maximum passage length
            
        Returns:
            list: Text passages
        """
        passages = []
        
        # First split by paragraphs (double newlines)
        paragraphs = content.split('\n\n')
        
        current_passage = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would exceed max_length
            if len(current_passage) + len(paragraph) > max_length:
                # Save current passage if it meets minimum length
                if len(current_passage) >= min_length:
                    passages.append(current_passage.strip())
                
                # Start new passage
                current_passage = paragraph
            else:
                # Add to current passage
                if current_passage:
                    current_passage += "\n\n" + paragraph
                else:
                    current_passage = paragraph
        
        # Don't forget the last passage
        if len(current_passage) >= min_length:
            passages.append(current_passage.strip())
        
        return passages
    
    def process_single_novel(self, file_path: Path, style_category: str) -> List[Dict]:
        """
        Process single novel file into training examples.
        
        Args:
            file_path: Path to novel file
            style_category: "modernist" or "travel"
            
        Returns:
            list: Generated training examples
        """
        self.logger.info(f"Processing {file_path.name}...")
        
        # Extract metadata and content
        metadata = self.load_novel_metadata(file_path)
        content = self.extract_novel_content(file_path)
        
        if not content:
            self.logger.warning(f"No content found in {file_path}")
            return []
        
        # Split into passages
        passages = self.split_into_passages(content)
        
        examples = []
        
        for i, passage in enumerate(passages):
            # Analyze sensory content
            sensory_analysis = self.analyzer.analyze_sensory_content(passage)
            
            # Skip passages with low sensory content
            if sensory_analysis['total_sensory_score'] < 0.2:
                continue
            
            # Extract detailed context
            context = self.context_extractor.extract_sensor_context(passage)
            
            # Map to sensor values
            sensor_data = self.sensor_mapper.map_to_sensor_values(context)
            confidence = self.sensor_mapper.calculate_mapping_confidence(context)
            
            # Skip low confidence mappings
            if confidence < 0.3:
                continue
            
            # Classify literary style
            style_scores = self.style_classifier.classify_passage(passage)
            
            # Create training example
            example = {
                'id': f"{file_path.stem}_passage_{i:03d}",
                'sensor_data': sensor_data,
                'target_paragraph': passage,
                'metadata': {
                    'source_file': file_path.name,
                    'style_category': style_category,
                    'style_scores': style_scores,
                    'sensory_analysis': sensory_analysis,
                    'context_extraction': context,
                    'mapping_confidence': confidence,
                    'novel_metadata': metadata,
                    'passage_index': i,
                    'generated_at': pd.Timestamp.now().isoformat()
                }
            }
            
            examples.append(example)
        
        self.logger.info(f"Generated {len(examples)} examples from {file_path.name}")
        return examples
    
    def process_novel_collection(self, 
                               max_files_per_category: Optional[int] = None,
                               min_confidence: float = 0.4) -> Dict[str, List[Dict]]:
        """
        Process entire novel collection.
        
        Args:
            max_files_per_category: Limit files per category
            min_confidence: Minimum confidence threshold
            
        Returns:
            dict: Categorized training examples
        """
        results = {
            'modernist': [],
            'travel': []
        }
        
        # Process Modernist novels
        if self.modernist_dir.exists():
            modernist_files = list(self.modernist_dir.glob("*.txt"))
            if max_files_per_category:
                modernist_files = modernist_files[:max_files_per_category]
                
            self.logger.info(f"Processing {len(modernist_files)} modernist novels...")
            
            for file_path in modernist_files:
                examples = self.process_single_novel(file_path, "modernist")
                # Filter by confidence
                high_conf_examples = [ex for ex in examples if ex['metadata']['mapping_confidence'] >= min_confidence]
                results['modernist'].extend(high_conf_examples)
        
        # Process Travel novels
        if self.travel_dir.exists():
            travel_files = list(self.travel_dir.glob("*.txt"))
            if max_files_per_category:
                travel_files = travel_files[:max_files_per_category]
                
            self.logger.info(f"Processing {len(travel_files)} travel novels...")
            
            for file_path in travel_files:
                examples = self.process_single_novel(file_path, "travel")
                # Filter by confidence
                high_conf_examples = [ex for ex in examples if ex['metadata']['mapping_confidence'] >= min_confidence]
                results['travel'].extend(high_conf_examples)
        
        # Log results
        total_examples = sum(len(examples) for examples in results.values())
        self.logger.info(f"Total generated examples: {total_examples}")
        self.logger.info(f"  Modernist: {len(results['modernist'])}")
        self.logger.info(f"  Travel: {len(results['travel'])}")
        
        return results
    
    def create_balanced_dataset(self, 
                              results: Dict[str, List[Dict]], 
                              target_size: int = 1000) -> List[Dict]:
        """
        Create balanced dataset from categorized results.
        
        Args:
            results: Categorized training examples
            target_size: Target dataset size
            
        Returns:
            list: Balanced dataset
        """
        balanced_dataset = []
        
        # Calculate samples per category
        categories = [cat for cat in results.keys() if results[cat]]
        samples_per_category = target_size // len(categories)
        
        for category in categories:
            examples = results[category]
            
            if len(examples) <= samples_per_category:
                # Use all examples
                balanced_dataset.extend(examples)
            else:
                # Sample randomly with quality weighting
                # Weight by confidence score
                weights = [ex['metadata']['mapping_confidence'] for ex in examples]
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                
                # Sample with replacement using weights
                selected_indices = np.random.choice(
                    len(examples), 
                    size=samples_per_category, 
                    replace=False,
                    p=weights
                )
                
                for idx in selected_indices:
                    balanced_dataset.append(examples[idx])
        
        # Shuffle final dataset
        np.random.shuffle(balanced_dataset)
        
        self.logger.info(f"Created balanced dataset with {len(balanced_dataset)} examples")
        return balanced_dataset
    
    def generate_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """Generate comprehensive dataset statistics."""
        
        stats = {
            'total_examples': len(dataset),
            'style_distribution': defaultdict(int),
            'confidence_stats': {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0
            },
            'sensor_stats': {
                'temperature': {'mean': 0, 'std': 0, 'range': (0, 0)},
                'humidity': {'mean': 0, 'std': 0, 'range': (0, 0)},
                'wind_direction': {'mean': 0, 'std': 0}
            },
            'passage_length_stats': {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0
            },
            'source_file_distribution': defaultdict(int)
        }
        
        if not dataset:
            return stats
        
        # Collect data for analysis
        confidences = []
        temperatures = []
        humidities = []
        wind_directions = []
        passage_lengths = []
        
        for example in dataset:
            # Style distribution
            style_category = example['metadata']['style_category']
            stats['style_distribution'][style_category] += 1
            
            # Source file distribution
            source_file = example['metadata']['source_file']
            stats['source_file_distribution'][source_file] += 1
            
            # Confidence
            confidence = example['metadata']['mapping_confidence']
            confidences.append(confidence)
            
            # Sensor data
            sensor_data = example['sensor_data']
            temperatures.append(sensor_data['temperature'])
            humidities.append(sensor_data['humidity'])
            wind_directions.append(sensor_data['wind_direction'])
            
            # Passage length
            passage_length = len(example['target_paragraph'])
            passage_lengths.append(passage_length)
        
        # Calculate statistics
        stats['confidence_stats'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        stats['sensor_stats'] = {
            'temperature': {
                'mean': np.mean(temperatures),
                'std': np.std(temperatures),
                'range': (np.min(temperatures), np.max(temperatures))
            },
            'humidity': {
                'mean': np.mean(humidities),
                'std': np.std(humidities),
                'range': (np.min(humidities), np.max(humidities))
            },
            'wind_direction': {
                'mean': np.mean(wind_directions),
                'std': np.std(wind_directions)
            }
        }
        
        stats['passage_length_stats'] = {
            'mean': np.mean(passage_lengths),
            'std': np.std(passage_lengths),
            'min': np.min(passage_lengths),
            'max': np.max(passage_lengths)
        }
        
        return stats
    
    def save_dataset(self, dataset: List[Dict], output_dir: str = "data/novel_dataset"):
        """Save dataset with train/val/test splits."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Shuffle dataset
        np.random.shuffle(dataset)
        
        # Create splits
        total_size = len(dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        
        splits = {
            'train': dataset[:train_size],
            'validation': dataset[train_size:train_size + val_size],
            'test': dataset[train_size + val_size:]
        }
        
        # Save each split
        for split_name, split_data in splits.items():
            split_path = output_path / f"{split_name}.json"
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(split_data)} examples to {split_path}")
        
        # Generate and save statistics
        stats = self.generate_dataset_statistics(dataset)
        stats_path = output_path / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset statistics saved to {stats_path}")
        
        return stats


def main():
    """Main execution function."""
    
    # Initialize processor
    processor = NovelCorpusProcessor()
    
    # Process novels (limit for testing)
    print("Processing novel collection...")
    results = processor.process_novel_collection(
        max_files_per_category=5,  # Limit for testing
        min_confidence=0.4
    )
    
    # Create balanced dataset
    print("Creating balanced dataset...")
    dataset = processor.create_balanced_dataset(results, target_size=500)
    
    # Save dataset
    print("Saving dataset...")
    stats = processor.save_dataset(dataset, "data/novel_sensor_dataset")
    
    # Print summary
    print("\n=== Dataset Generation Summary ===")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Style distribution: {dict(stats['style_distribution'])}")
    print(f"Average confidence: {stats['confidence_stats']['mean']:.3f}")
    print(f"Temperature range: {stats['sensor_stats']['temperature']['range']}")
    print(f"Average passage length: {stats['passage_length_stats']['mean']:.0f} characters")
    
    # Show example
    if dataset:
        example = dataset[0]
        print(f"\n=== Sample Generated Data ===")
        print(f"Source: {example['metadata']['source_file']}")
        print(f"Style: {example['metadata']['style_category']}")
        print(f"Temperature: {example['sensor_data']['temperature']}°C")
        print(f"Humidity: {example['sensor_data']['humidity']}%")
        print(f"Confidence: {example['metadata']['mapping_confidence']:.3f}")
        print(f"Passage preview: {example['target_paragraph'][:200]}...")


if __name__ == "__main__":
    main()