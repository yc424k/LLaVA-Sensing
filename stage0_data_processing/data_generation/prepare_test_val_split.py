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

def create_test_val_split(samples, genre_name, total_count, output_dir):
    """
    Select samples and split into test/validation sets
    
    Args:
        samples (list): List of all samples for this genre
        genre_name (str): Genre name ('travel' or 'modernist')
        total_count (int): Total number of samples to select (30k)
        output_dir (str): Output directory
    
    Returns:
        tuple: (test_count, val_count)
    """
    if len(samples) < total_count:
        print(f"Warning: {genre_name} has only {len(samples)} samples, less than target {total_count}")
        total_count = len(samples)
    
    # Randomly sample total_count samples
    print(f"Randomly selecting {total_count:,} out of {len(samples):,} {genre_name} samples...")
    random.shuffle(samples)
    selected_samples = samples[:total_count]
    
    # Split into test (first 15k) and validation (last 15k)
    split_point = total_count // 2
    test_samples = selected_samples[:split_point]
    val_samples = selected_samples[split_point:]
    
    print(f"Splitting {genre_name}: {len(test_samples):,} test + {len(val_samples):,} validation")
    
    # Save test set
    test_count = save_split(test_samples, genre_name, "test", output_dir)
    
    # Save validation set
    val_count = save_split(val_samples, genre_name, "validation", output_dir)
    
    return test_count, val_count

def save_split(samples, genre_name, split_name, output_dir):
    """
    Save a split (test or validation) to chunk files
    
    Args:
        samples (list): List of samples to save
        genre_name (str): Genre name
        split_name (str): 'test' or 'validation'
        output_dir (str): Output directory
    
    Returns:
        int: Number of samples saved
    """
    # Create output directory
    split_output_dir = os.path.join(output_dir, f"{genre_name}_{split_name}")
    os.makedirs(split_output_dir, exist_ok=True)
    
    # Split into chunks (100 samples per file)
    chunk_size = 100
    chunk_num = 1
    
    print(f"Saving {len(samples):,} {genre_name} {split_name} samples in chunks of {chunk_size}...")
    
    for i in range(0, len(samples), chunk_size):
        chunk_data = samples[i:i + chunk_size]
        chunk_filename = f"{genre_name}_{split_name}_chunk_{chunk_num:03d}.json"
        chunk_path = os.path.join(split_output_dir, chunk_filename)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        chunk_num += 1
    
    # Create index file
    index_data = {
        "total_samples": len(samples),
        "genre": genre_name,
        "split": split_name,
        "selection_method": "random",
        "random_seed": 42,
        "chunks": chunk_num - 1,
        "chunk_size": chunk_size,
        "source_dataset": "hybrid_novel_dataset_chunks (processed)",
        "files": [f"{genre_name}_{split_name}_chunk_{i:03d}.json" for i in range(1, chunk_num)]
    }
    
    index_path = os.path.join(split_output_dir, f"{genre_name}_{split_name}_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {chunk_num-1} chunk files and index for {genre_name} {split_name}")
    return len(samples)

def create_overall_statistics(output_dir, results):
    """
    Create overall statistics file
    
    Args:
        results (dict): Results from processing
    """
    stats = {
        "dataset_split_summary": {
            "total_samples": sum(results[genre]['test'] + results[genre]['val'] for genre in results),
            "random_seed": 42,
            "split_method": "50-50 test-validation per genre",
            "genres": {}
        },
        "file_structure": {
            "chunk_size": 100,
            "directories": []
        }
    }
    
    for genre in results:
        stats["dataset_split_summary"]["genres"][genre] = {
            "test_samples": results[genre]['test'],
            "validation_samples": results[genre]['val'], 
            "total_samples": results[genre]['test'] + results[genre]['val']
        }
        
        # Add directory info
        stats["file_structure"]["directories"].extend([
            f"{genre}_test/",
            f"{genre}_validation/"
        ])
    
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Created dataset statistics: {stats_path}")

def main():
    input_dir = "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/hybrid_novel_dataset_chunks"
    output_dir = "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/test_val_30k_each"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Test/Validation Split Tool - 30,000 samples per genre")
    print("=" * 65)
    
    # Collect samples by genre
    print("Step 1: Collecting all samples by genre...")
    samples_by_genre = collect_all_samples_by_genre(input_dir)
    
    print(f"\nGenre distribution in source dataset:")
    print(f"Travel: {len(samples_by_genre['travel']):,} samples")
    print(f"Modernist: {len(samples_by_genre['modernist']):,} samples")
    print(f"Total: {len(samples_by_genre['travel']) + len(samples_by_genre['modernist']):,} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each genre
    target_count = 30000
    results = {}
    
    print(f"\nStep 2: Creating test/validation splits for each genre...")
    print(f"Target: {target_count:,} samples per genre (15k test + 15k validation)")
    
    for genre in ['travel', 'modernist']:
        print(f"\nProcessing {genre} genre...")
        test_count, val_count = create_test_val_split(
            samples_by_genre[genre], 
            genre, 
            target_count, 
            output_dir
        )
        results[genre] = {'test': test_count, 'val': val_count}
    
    # Create overall statistics
    print(f"\nStep 3: Creating overall statistics...")
    create_overall_statistics(output_dir, results)
    
    print(f"\nDataset preparation completed!")
    print(f"=" * 65)
    
    for genre in results:
        print(f"{genre.capitalize()} dataset:")
        print(f"  Test: {results[genre]['test']:,} samples")
        print(f"  Validation: {results[genre]['val']:,} samples")
        print(f"  Total: {results[genre]['test'] + results[genre]['val']:,} samples")
    
    total_samples = sum(results[genre]['test'] + results[genre]['val'] for genre in results)
    print(f"\nGrand total: {total_samples:,} samples")
    print(f"Output directory: {output_dir}")
    
    # Verify selections
    print(f"\nSelection rates:")
    for genre in results:
        total_selected = results[genre]['test'] + results[genre]['val']
        selection_rate = total_selected / len(samples_by_genre[genre]) * 100
        print(f"{genre.capitalize()}: {selection_rate:.1f}% ({total_selected:,}/{len(samples_by_genre[genre]):,})")

if __name__ == "__main__":
    main()
