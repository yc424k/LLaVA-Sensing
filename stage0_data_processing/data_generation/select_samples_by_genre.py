import json
import glob
import os
import random
from pathlib import Path

def collect_all_samples_by_genre(input_dir):
    """
    Collect all samples from processed hybrid dataset and group by genre
    
    Returns:
        dict: {"travel": [...], "modernist": [...]}
    """
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    samples_by_genre = {"travel": [], "modernist": []}
    
    print(f"Collecting samples from {len(json_files)} files...")
    
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
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(json_files)} files...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return samples_by_genre

def randomly_select_and_save(samples, genre_name, target_count, output_dir):
    """
    Randomly select target_count items and save to separate files
    
    Args:
        samples (list): List of all samples for this genre
        genre_name (str): Genre name ('travel' or 'modernist')
        target_count (int): Number of samples to select
        output_dir (str): Output directory
    
    Returns:
        int: Number of samples actually selected
    """
    if len(samples) < target_count:
        print(f"Warning: {genre_name} has only {len(samples)} samples, less than target {target_count}")
        target_count = len(samples)
    
    # Randomly sample
    print(f"Randomly selecting {target_count} out of {len(samples)} {genre_name} samples...")
    random.shuffle(samples)
    selected_samples = samples[:target_count]
    
    # Create output directory
    genre_output_dir = os.path.join(output_dir, f"{genre_name}_selected_15k")
    os.makedirs(genre_output_dir, exist_ok=True)
    
    # Split into chunks (100 samples per file)
    chunk_size = 100
    chunk_num = 1
    
    print(f"Saving {len(selected_samples)} {genre_name} samples in chunks of {chunk_size}...")
    
    for i in range(0, len(selected_samples), chunk_size):
        chunk_data = selected_samples[i:i + chunk_size]
        chunk_filename = f"{genre_name}_selected_chunk_{chunk_num:03d}.json"
        chunk_path = os.path.join(genre_output_dir, chunk_filename)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        chunk_num += 1
    
    # Create index file
    index_data = {
        "total_samples": len(selected_samples),
        "genre": genre_name,
        "selection_method": "random",
        "random_seed": 42,
        "chunks": chunk_num - 1,
        "chunk_size": chunk_size,
        "source_dataset": "hybrid_novel_dataset_chunks (processed)",
        "source_total_samples": len(samples),
        "files": [f"{genre_name}_selected_chunk_{i:03d}.json" for i in range(1, chunk_num)]
    }
    
    index_path = os.path.join(genre_output_dir, f"{genre_name}_selected_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {chunk_num-1} chunk files and index for {genre_name}")
    return len(selected_samples)

def create_combined_statistics(output_dir, travel_count, modernist_count):
    """
    Create combined statistics file
    """
    stats = {
        "selection_summary": {
            "travel_samples": travel_count,
            "modernist_samples": modernist_count,
            "total_samples": travel_count + modernist_count,
            "selection_method": "random_sampling",
            "random_seed": 42,
            "target_per_genre": 15000
        },
        "file_structure": {
            "travel_directory": "travel_selected_15k/",
            "modernist_directory": "modernist_selected_15k/",
            "chunk_size": 100,
            "travel_chunks": (travel_count + 99) // 100,  # ceiling division
            "modernist_chunks": (modernist_count + 99) // 100
        }
    }
    
    stats_path = os.path.join(output_dir, "selection_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Created selection statistics: {stats_path}")

def main():
    input_dir = "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/hybrid_novel_dataset_chunks"
    output_dir = "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/selected_15k_each"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Random Selection Tool - 15,000 samples per genre")
    print("=" * 60)
    
    # Collect samples by genre
    print("Step 1: Collecting all samples by genre...")
    samples_by_genre = collect_all_samples_by_genre(input_dir)
    
    print(f"\nGenre distribution in source dataset:")
    print(f"Travel: {len(samples_by_genre['travel']):,} samples")
    print(f"Modernist: {len(samples_by_genre['modernist']):,} samples")
    print(f"Total: {len(samples_by_genre['travel']) + len(samples_by_genre['modernist']):,} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select and save each genre
    target_count = 15000
    
    print(f"\nStep 2: Randomly selecting {target_count:,} samples for each genre...")
    
    travel_count = randomly_select_and_save(
        samples_by_genre['travel'], 
        'travel', 
        target_count, 
        output_dir
    )
    
    modernist_count = randomly_select_and_save(
        samples_by_genre['modernist'], 
        'modernist', 
        target_count, 
        output_dir
    )
    
    # Create combined statistics
    print(f"\nStep 3: Creating statistics...")
    create_combined_statistics(output_dir, travel_count, modernist_count)
    
    print(f"\nSelection completed!")
    print(f"Travel dataset: {travel_count:,} samples")
    print(f"Modernist dataset: {modernist_count:,} samples")
    print(f"Total selected: {travel_count + modernist_count:,} samples")
    print(f"Output directory: {output_dir}")
    
    # Verify selection
    print(f"\nVerification:")
    print(f"Travel selection rate: {travel_count/len(samples_by_genre['travel'])*100:.1f}%")
    print(f"Modernist selection rate: {modernist_count/len(samples_by_genre['modernist'])*100:.1f}%")

if __name__ == "__main__":
    main()
