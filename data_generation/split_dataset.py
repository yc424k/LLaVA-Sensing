#!/usr/bin/env python3
"""
Split large JSON dataset into smaller chunks for better handling.
"""

import json
import os
import math
from pathlib import Path


def split_json_dataset(input_file, output_dir="data/chunks", chunk_size=100):
    """
    Split large JSON dataset into smaller chunks.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save chunks
        chunk_size: Number of examples per chunk
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load original dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_examples = len(data)
    total_chunks = math.ceil(total_examples / chunk_size)
    
    print(f"Dataset info:")
    print(f"  Total examples: {total_examples}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Output directory: {output_path}")
    
    # Split into chunks
    chunk_info = []
    
    for i in range(0, total_examples, chunk_size):
        chunk_num = i // chunk_size + 1
        chunk_data = data[i:i + chunk_size]
        
        # Create chunk filename
        chunk_filename = f"novel_dataset_chunk_{chunk_num:03d}.json"
        chunk_path = output_path / chunk_filename
        
        # Save chunk
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        # Record chunk info
        chunk_info.append({
            "chunk_number": chunk_num,
            "filename": chunk_filename,
            "start_index": i,
            "end_index": min(i + chunk_size, total_examples),
            "size": len(chunk_data),
            "file_path": str(chunk_path)
        })
        
        print(f"  Created {chunk_filename}: {len(chunk_data)} examples")
    
    # Create index file
    index_info = {
        "original_file": str(input_file),
        "total_examples": total_examples,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "chunks": chunk_info,
        "split_timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    index_path = output_path / "chunks_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)
    
    print(f"\nSplit complete!")
    print(f"  Index file: {index_path}")
    print(f"  Chunk files: {total_chunks} files in {output_path}")
    
    return chunk_info


def create_split_reader():
    """Create a utility to read split datasets."""
    
    reader_code = '''#!/usr/bin/env python3
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
'''
    
    with open("/home/yc424k/LLaVA-NeXT/data_generation/dataset_reader.py", 'w') as f:
        f.write(reader_code)
    
    print("Created dataset_reader.py utility")


def main():
    """Main function to split the dataset."""
    
    input_file = "/home/yc424k/LLaVA-NeXT/data_generation/data/novel_dataset.json"
    output_dir = "/home/yc424k/LLaVA-NeXT/data_generation/data/chunks"
    chunk_size = 100
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Split dataset
    chunk_info = split_json_dataset(input_file, output_dir, chunk_size)
    
    # Create reader utility
    create_split_reader()
    
    print("\n" + "="*50)
    print("Dataset splitting complete!")
    print(f"Original file: {input_file}")
    print(f"Chunks directory: {output_dir}")
    print(f"Total chunks: {len(chunk_info)}")
    print(f"Chunk size: {chunk_size} examples each")
    print("\nTo use the chunks:")
    print("  python3 dataset_reader.py")


if __name__ == "__main__":
    main()