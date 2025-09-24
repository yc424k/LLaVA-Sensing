import json
import glob
import os
import math
from pathlib import Path

def radians_to_compass_value(radians):
    """
    Convert radians (0 to 2π) to compass direction value (0-16)
    
    Args:
        radians (float): Wind direction in radians (0 to 2π)
    
    Returns:
        int: Direction value (0-16) according to the table
    """
    
    # Normalize radians to 0-2π range
    radians = radians % (2 * math.pi)
    
    # Convert to degrees for easier calculation
    degrees = math.degrees(radians)
    
    # Calculate direction index (each direction covers 22.5 degrees)
    # Round to nearest integer and ensure it's in 0-15 range
    direction_index = round(degrees / 22.5) % 16
    
    # Map to 0-16 range (16 is same as 0 - North)
    if direction_index == 0:
        return 0  # North
    else:
        return direction_index

def get_direction_name(value):
    """Get direction name from value (0-16)"""
    directions = [
        "North",                    # 0
        "Northeast by north",       # 1
        "Northeast",               # 2
        "Northeast by east",       # 3
        "East",                    # 4
        "Southeast by east",       # 5
        "Southeast",               # 6
        "Southeast by south",      # 7
        "South",                   # 8
        "Southwest by south",      # 9
        "Southwest",               # 10
        "Southwest by west",       # 11
        "West",                    # 12
        "Northwest by west",       # 13
        "Northwest",               # 14
        "Northwest by north",      # 15
        "North"                    # 16 (same as 0)
    ]
    
    return directions[min(value, 16)]

def convert_json_files(input_dir, output_dir=None):
    """
    Convert all wind_direction values in JSON files from radians to compass values
    
    Args:
        input_dir (str): Directory containing JSON files
        output_dir (str): Output directory (if None, overwrites original files)
    """
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    total_converted = 0
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for i, file_path in enumerate(json_files):
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_converted = 0
            
            # Convert wind_direction values
            for item in data:
                if 'sensor_data' in item and 'wind_direction' in item['sensor_data']:
                    original_radians = item['sensor_data']['wind_direction']
                    compass_value = radians_to_compass_value(original_radians)
                    
                    # Replace with compass value
                    item['sensor_data']['wind_direction'] = compass_value
                    file_converted += 1
                    total_converted += 1
            
            # Determine output path
            if output_dir:
                output_path = os.path.join(output_dir, os.path.basename(file_path))
            else:
                output_path = file_path
            
            # Write converted file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"[{i+1:3d}/{len(json_files)}] {os.path.basename(file_path)}: {file_converted} values converted")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"\nTotal conversions completed: {total_converted}")
    return total_converted

def preview_conversions(input_dir, num_samples=10):
    """
    Preview what the conversions would look like
    """
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    print("Preview of wind_direction conversions:")
    print("=" * 80)
    print(f"{'Original (radians)':>18} {'Degrees':>10} {'Value':>6} {'Direction':>20}")
    print("-" * 80)
    
    count = 0
    for file_path in json_files[:3]:  # Check first 3 files
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                if count >= num_samples:
                    break
                    
                if 'sensor_data' in item and 'wind_direction' in item['sensor_data']:
                    radians = item['sensor_data']['wind_direction']
                    degrees = math.degrees(radians)
                    value = radians_to_compass_value(radians)
                    direction = get_direction_name(value)
                    
                    print(f"{radians:18.3f} {degrees:10.1f} {value:6d} {direction:>20}")
                    count += 1
            
            if count >= num_samples:
                break
                
        except Exception as e:
            print(f"Error in preview: {e}")
            continue
    
    print("=" * 80)

if __name__ == "__main__":
    input_directory = "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/hybrid_novel_dataset_chunks"
    
    print("Wind Direction Conversion Tool")
    print("=" * 50)
    
    # Show preview first
    preview_conversions(input_directory, 15)
    
    # Proceed with conversion automatically
    print("\nProceeding with conversion...")
    
    if True:
        # Create backup directory
        backup_dir = input_directory + "_backup"
        print(f"\nCreating backup in: {backup_dir}")
        
        # Copy files to backup first
        import shutil
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(input_directory, backup_dir)
        
        # Convert files
        total = convert_json_files(input_directory)
        print(f"\nConversion completed! {total} wind_direction values converted.")
        print(f"Original files backed up to: {backup_dir}")
    else:
        print("Conversion cancelled.")
