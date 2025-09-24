import json
import glob
import os
import re
from pathlib import Path

def smart_truncate_paragraph(text, max_length=5000, preserve_sentences=True):
    """
    Intelligently truncate paragraph while preserving readability
    
    Args:
        text (str): Original paragraph text
        max_length (int): Maximum character length
        preserve_sentences (bool): Try to cut at sentence boundaries
    
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if preserve_sentences:
        # Try to cut at sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        result = ""
        
        for sentence in sentences:
            if len(result + sentence) <= max_length - 3:  # Save space for "..."
                result += sentence + ". "
            else:
                break
        
        if result:
            return result.rstrip() + "..."
    
    # Fallback: simple truncation with word boundary
    truncated = text[:max_length-3]
    # Try to cut at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If word boundary is not too far back
        truncated = truncated[:last_space]
    
    return truncated + "..."

def split_long_paragraph(text, target_length=3000, overlap=200):
    """
    Split extremely long paragraphs into multiple samples
    
    Args:
        text (str): Original paragraph text
        target_length (int): Target length for each chunk
        overlap (int): Overlap between chunks for context
    
    Returns:
        list: List of text chunks
    """
    if len(text) <= target_length * 1.5:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + target_length
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Try to find sentence boundary
        chunk_text = text[start:end]
        last_sentence = max(
            chunk_text.rfind('. '),
            chunk_text.rfind('! '),
            chunk_text.rfind('? ')
        )
        
        if last_sentence > target_length * 0.7:
            # Good sentence boundary found
            actual_end = start + last_sentence + 2
            chunks.append(text[start:actual_end])
            start = actual_end - overlap
        else:
            # No good sentence boundary, use word boundary
            last_space = chunk_text.rfind(' ')
            if last_space > target_length * 0.8:
                actual_end = start + last_space
                chunks.append(text[start:actual_end])
                start = actual_end - overlap
            else:
                # Force cut
                chunks.append(chunk_text)
                start = end - overlap
        
        # Avoid infinite loop
        if start < 0:
            start = 0
    
    return [chunk for chunk in chunks if chunk.strip()]

def analyze_and_fix_dataset(input_dirs, max_length=5000, split_threshold=15000):
    """
    Analyze and fix paragraph lengths in dataset
    
    Args:
        input_dirs (list): List of input directories
        max_length (int): Maximum length for truncation
        split_threshold (int): Threshold for splitting vs truncating
    """
    
    stats = {
        'total_processed': 0,
        'truncated': 0,
        'split': 0,
        'unchanged': 0,
        'new_samples_created': 0
    }
    
    for input_dir in input_dirs:
        print(f"\nProcessing {input_dir}...")
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        
        for file_path in json_files:
            if file_path.endswith('_index.json'):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                new_data = []
                
                for item in data:
                    stats['total_processed'] += 1
                    paragraph = item.get('target_paragraph', '')
                    
                    if len(paragraph) <= max_length:
                        new_data.append(item)
                        stats['unchanged'] += 1
                    
                    elif len(paragraph) > split_threshold:
                        # Split into multiple samples
                        chunks = split_long_paragraph(paragraph, target_length=max_length//2)
                        
                        for i, chunk in enumerate(chunks):
                            new_item = item.copy()
                            new_item['target_paragraph'] = chunk
                            new_item['id'] = f"{item['id']}_part_{i+1}"
                            
                            # Update metadata to indicate this is a split paragraph
                            if 'metadata' not in new_item:
                                new_item['metadata'] = {}
                            new_item['metadata']['split_from_original'] = True
                            new_item['metadata']['part_number'] = i + 1
                            new_item['metadata']['total_parts'] = len(chunks)
                            new_item['metadata']['original_length'] = len(paragraph)
                            
                            new_data.append(new_item)
                        
                        stats['split'] += 1
                        stats['new_samples_created'] += len(chunks) - 1
                    
                    else:
                        # Truncate
                        item['target_paragraph'] = smart_truncate_paragraph(paragraph, max_length)
                        
                        # Update metadata to indicate truncation
                        if 'metadata' not in item:
                            item['metadata'] = {}
                        item['metadata']['truncated'] = True
                        item['metadata']['original_length'] = len(paragraph)
                        
                        new_data.append(item)
                        stats['truncated'] += 1
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, indent=2, ensure_ascii=False)
                
                print(f"  {os.path.basename(file_path)}: {len(data)} -> {len(new_data)} samples")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    return stats

def create_length_distribution_report(input_dirs, output_file):
    """Create a report of paragraph length distribution"""
    
    all_lengths = []
    
    for input_dir in input_dirs:
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        
        for file_path in json_files:
            if file_path.endswith('_index.json'):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for item in data:
                    paragraph = item.get('target_paragraph', '')
                    all_lengths.append(len(paragraph))
                    
            except Exception as e:
                continue
    
    all_lengths.sort()
    
    report = {
        "total_samples": len(all_lengths),
        "statistics": {
            "mean": sum(all_lengths) / len(all_lengths),
            "median": all_lengths[len(all_lengths)//2],
            "min": min(all_lengths),
            "max": max(all_lengths),
            "percentiles": {
                "50th": all_lengths[int(len(all_lengths) * 0.5)],
                "75th": all_lengths[int(len(all_lengths) * 0.75)],
                "90th": all_lengths[int(len(all_lengths) * 0.9)],
                "95th": all_lengths[int(len(all_lengths) * 0.95)],
                "99th": all_lengths[int(len(all_lengths) * 0.99)]
            }
        },
        "length_bins": {}
    }
    
    # Create length bins
    bins = [0, 1000, 2000, 5000, 10000, 20000, 50000, float('inf')]
    bin_labels = ["0-1K", "1K-2K", "2K-5K", "5K-10K", "10K-20K", "20K-50K", "50K+"]
    
    for i, label in enumerate(bin_labels):
        count = sum(1 for length in all_lengths if bins[i] <= length < bins[i+1])
        report["length_bins"][label] = {
            "count": count,
            "percentage": (count / len(all_lengths)) * 100
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

def main():
    # Configuration
    MAX_LENGTH = 5000  # Maximum length for any paragraph
    SPLIT_THRESHOLD = 15000  # Split paragraphs longer than this
    
    input_dirs = [
        "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/genre_split/travel_dataset",
        "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/genre_split/modernist_dataset"
    ]
    
    print("Paragraph Length Normalization Tool")
    print("=" * 50)
    print(f"Max length: {MAX_LENGTH} characters")
    print(f"Split threshold: {SPLIT_THRESHOLD} characters")
    
    # Create before report
    print("\nCreating 'before' report...")
    before_report = create_length_distribution_report(
        input_dirs, 
        "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/paragraph_lengths_before.json"
    )
    
    print(f"Before processing:")
    print(f"  Total samples: {before_report['total_samples']}")
    print(f"  Mean length: {before_report['statistics']['mean']:.1f}")
    print(f"  Max length: {before_report['statistics']['max']}")
    print(f"  95th percentile: {before_report['statistics']['percentiles']['95th']}")
    
    # Process datasets
    print(f"\nProcessing datasets...")
    stats = analyze_and_fix_dataset(input_dirs, MAX_LENGTH, SPLIT_THRESHOLD)
    
    # Create after report
    print("\nCreating 'after' report...")
    after_report = create_length_distribution_report(
        input_dirs,
        "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/paragraph_lengths_after.json"
    )
    
    print(f"\nProcessing completed!")
    print(f"Total samples processed: {stats['total_processed']}")
    print(f"Unchanged: {stats['unchanged']}")
    print(f"Truncated: {stats['truncated']}")
    print(f"Split: {stats['split']}")
    print(f"New samples created: {stats['new_samples_created']}")
    
    print(f"\nAfter processing:")
    print(f"  Total samples: {after_report['total_samples']}")
    print(f"  Mean length: {after_report['statistics']['mean']:.1f}")
    print(f"  Max length: {after_report['statistics']['max']}")
    print(f"  95th percentile: {after_report['statistics']['percentiles']['95th']}")

if __name__ == "__main__":
    main()
