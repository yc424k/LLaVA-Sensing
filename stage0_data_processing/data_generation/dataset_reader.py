#!/usr/bin/env python3
"""
Utility to read split dataset chunks.
"""

import json
import os
from pathlib import Path


class ChunkedDatasetReader:
    """Reader for chunked datasets."""
    
    def __init__(self, chunks_dir="data/chunks"):
        self.chunks_dir = Path(chunks_dir)
        self.index_file = self.chunks_dir / "chunks_index.json"
        self.index = None
        self.load_index()
    
    def load_index(self):
        """Load chunk index information."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
    
    def get_chunk(self, chunk_number):
        """Load specific chunk by number."""
        if not self.index:
            raise ValueError("Index not loaded")
        
        chunk_info = next((c for c in self.index['chunks'] if c['chunk_number'] == chunk_number), None)
        if not chunk_info:
            raise ValueError(f"Chunk {chunk_number} not found")
        
        chunk_path = self.chunks_dir / chunk_info['filename']
        with open(chunk_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_chunks(self):
        """Generator to iterate through all chunks."""
        if not self.index:
            raise ValueError("Index not loaded")
        
        for chunk_info in self.index['chunks']:
            chunk_path = self.chunks_dir / chunk_info['filename']
            with open(chunk_path, 'r', encoding='utf-8') as f:
                yield json.load(f)
    
    def get_example_by_global_index(self, global_index):
        """Get example by its original position in the dataset."""
        if not self.index:
            raise ValueError("Index not loaded")
        
        chunk_size = self.index['chunk_size']
        chunk_number = (global_index // chunk_size) + 1
        local_index = global_index % chunk_size
        
        chunk_data = self.get_chunk(chunk_number)
        if local_index < len(chunk_data):
            return chunk_data[local_index]
        else:
            raise IndexError(f"Global index {global_index} out of range")
    
    def get_info(self):
        """Get dataset information."""
        if not self.index:
            return None
        
        return {
            'total_examples': self.index['total_examples'],
            'total_chunks': self.index['total_chunks'],
            'chunk_size': self.index['chunk_size'],
            'original_file': self.index['original_file']
        }
    
    def search_by_style(self, style_category):
        """Find examples by style category."""
        results = []
        
        for chunk_data in self.get_all_chunks():
            for example in chunk_data:
                if example['metadata']['style_category'] == style_category:
                    results.append(example)
        
        return results
    
    def get_statistics(self):
        """Get dataset statistics."""
        stats = {
            'total_examples': 0,
            'style_distribution': {},
            'temperature_range': [float('inf'), float('-inf')],
            'humidity_range': [float('inf'), float('-inf')]
        }
        
        for chunk_data in self.get_all_chunks():
            stats['total_examples'] += len(chunk_data)
            
            for example in chunk_data:
                # Style distribution
                style = example['metadata']['style_category']
                stats['style_distribution'][style] = stats['style_distribution'].get(style, 0) + 1
                
                # Temperature range
                temp = example['sensor_data']['temperature']
                stats['temperature_range'][0] = min(stats['temperature_range'][0], temp)
                stats['temperature_range'][1] = max(stats['temperature_range'][1], temp)
                
                # Humidity range
                humidity = example['sensor_data']['humidity']
                stats['humidity_range'][0] = min(stats['humidity_range'][0], humidity)
                stats['humidity_range'][1] = max(stats['humidity_range'][1], humidity)
        
        return stats


def main():
    """Example usage of ChunkedDatasetReader."""
    reader = ChunkedDatasetReader()
    
    # Get dataset info
    info = reader.get_info()
    print(f"Dataset Info: {info}")
    
    # Get first chunk
    first_chunk = reader.get_chunk(1)
    print(f"First chunk size: {len(first_chunk)}")
    
    # Get specific example
    example = reader.get_example_by_global_index(0)
    print(f"First example ID: {example['id']}")
    
    # Get statistics
    stats = reader.get_statistics()
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
