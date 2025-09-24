import json
import glob
import os
import random
from pathlib import Path

def collect_samples_by_genre(input_dir):
    """
    Collect all samples from JSON files and group by genre
    
    Returns:
        dict: {"travel": [...], "modernist": [...]}
    """
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    samples_by_genre = {"travel": [], "modernist": []}
    
    print(f"Processing {len(json_files)} files...")
    
    for i, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                # Extract genre from metadata.style_category
                metadata = item.get('metadata', {})
                style_category = metadata.get('style_category', '')
                
                if style_category in ['travel', 'modernist']:
                    samples_by_genre[style_category].append(item)
                else:
                    # Fallback to other methods if style_category is not found
                    genre = analyze_sample_for_genre(item)
                    samples_by_genre[genre].append(item)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(json_files)} files...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return samples_by_genre

def sample_and_save_genre(samples, genre_name, target_count, output_dir):
    """
    Randomly sample target_count items and save to separate files
    """
    if len(samples) < target_count:
        print(f"Warning: {genre_name} has only {len(samples)} samples, less than target {target_count}")
        target_count = len(samples)
    
    # Randomly sample
    random.shuffle(samples)
    selected_samples = samples[:target_count]
    
    # Create output directory
    genre_output_dir = os.path.join(output_dir, f"{genre_name}_dataset")
    os.makedirs(genre_output_dir, exist_ok=True)
    
    # Split into chunks (similar to original structure)
    chunk_size = 100  # 100 samples per file
    chunk_num = 1
    
    for i in range(0, len(selected_samples), chunk_size):
        chunk_data = selected_samples[i:i + chunk_size]
        chunk_filename = f"{genre_name}_dataset_chunk_{chunk_num:03d}.json"
        chunk_path = os.path.join(genre_output_dir, chunk_filename)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunk_data)} samples to {chunk_filename}")
        chunk_num += 1
    
    # Create index file
    index_data = {
        "total_samples": len(selected_samples),
        "genre": genre_name,
        "chunks": chunk_num - 1,
        "chunk_size": chunk_size,
        "files": [f"{genre_name}_dataset_chunk_{i:03d}.json" for i in range(1, chunk_num)]
    }
    
    index_path = os.path.join(genre_output_dir, f"{genre_name}_dataset_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    return len(selected_samples)

def analyze_sample_for_genre(sample):
    """
    Advanced genre detection based on multiple fields
    """
    # Check ID for genre indicators
    item_id = sample.get('id', '').lower()
    if 'travel' in item_id:
        return 'travel'
    elif 'modernist' in item_id:
        return 'modernist'
    
    # Check conversations
    conversations = sample.get('conversations', [])
    for conv in conversations:
        if isinstance(conv, dict):
            value = conv.get('value', '').lower()
            if any(keyword in value for keyword in ['여행', 'travel', 'journey', '길', 'road', 'path']):
                return 'travel'
            elif any(keyword in value for keyword in ['모더니즘', 'modernist', '의식의 흐름', 'stream of consciousness']):
                return 'modernist'
    
    # Check response content
    response = sample.get('response', '').lower()
    if any(keyword in response for keyword in ['여행', 'travel', 'journey', '길', 'road']):
        return 'travel'
    
    # Default to modernist if uncertain
    return 'modernist'

def main():
    input_dir = "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/hybrid_novel_dataset_chunks"
    output_dir = "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/genre_split"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Starting genre-based dataset splitting...")
    print("=" * 60)
    
    # Collect samples by genre
    print("Collecting samples by genre...")
    samples_by_genre = collect_samples_by_genre(input_dir)
    
    print(f"\nGenre distribution:")
    print(f"Travel: {len(samples_by_genre['travel'])} samples")
    print(f"Modernist: {len(samples_by_genre['modernist'])} samples")
    print(f"Total: {len(samples_by_genre['travel']) + len(samples_by_genre['modernist'])} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample and save each genre
    target_count = 10000
    
    print(f"\nSampling {target_count} samples for each genre...")
    
    travel_count = sample_and_save_genre(
        samples_by_genre['travel'], 
        'travel', 
        target_count, 
        output_dir
    )
    
    modernist_count = sample_and_save_genre(
        samples_by_genre['modernist'], 
        'modernist', 
        target_count, 
        output_dir
    )
    
    print(f"\nSplit completed!")
    print(f"Travel dataset: {travel_count} samples")
    print(f"Modernist dataset: {modernist_count} samples")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
