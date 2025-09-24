import math

def radians_to_compass_direction(radians):
    """
    Convert radians (0 to 2π) to 16-point compass direction
    
    Args:
        radians (float): Wind direction in radians (0 to 2π)
    
    Returns:
        tuple: (direction_index, direction_name)
    """
    
    # Normalize radians to 0-2π range
    radians = radians % (2 * math.pi)
    
    # Convert to degrees for easier calculation
    degrees = math.degrees(radians)
    
    # 16 compass directions (22.5 degrees each)
    directions = [
        "North",           # 0
        "North-Northeast", # 1 (Northeast by north)
        "Northeast",       # 2
        "East-Northeast",  # 3 (Northeast by east)
        "East",           # 4
        "East-Southeast", # 5 (Southeast by east)
        "Southeast",      # 6
        "South-Southeast", # 7 (Southeast by south)
        "South",          # 8
        "South-Southwest", # 9 (Southwest by south)
        "Southwest",      # 10
        "West-Southwest", # 11 (Southwest by west)
        "West",           # 12
        "West-Northwest", # 13 (Northwest by west)
        "Northwest",      # 14
        "North-Northwest" # 15 (Northwest by north)
    ]
    
    # Calculate direction index (each direction covers 22.5 degrees)
    direction_index = round(degrees / 22.5) % 16
    
    return direction_index, directions[direction_index]

def get_wind_direction_stats(json_files):
    """
    Analyze wind direction distribution in dataset files
    """
    import json
    
    direction_counts = {i: 0 for i in range(16)}
    total_count = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for item in data:
                if 'sensor_data' in item and 'wind_direction' in item['sensor_data']:
                    radians = item['sensor_data']['wind_direction']
                    direction_index, _ = radians_to_compass_direction(radians)
                    direction_counts[direction_index] += 1
                    total_count += 1
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return direction_counts, total_count

# Example usage and testing
if __name__ == "__main__":
    # Test with some sample values
    test_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.283]
    
    print("Radians -> Direction Conversion:")
    print("=" * 50)
    for radians in test_values:
        index, direction = radians_to_compass_direction(radians)
        degrees = math.degrees(radians)
        print(f"{radians:.3f} rad ({degrees:.1f}°) -> {index:2d}: {direction}")
    
    # Test with your dataset
    import glob
    json_files = glob.glob(
        "/home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation/data/processed/hybrid_novel_dataset_chunks/*.json"
    )
    
    if json_files:
        print(f"\nAnalyzing {len(json_files)} dataset files...")
        direction_counts, total = get_wind_direction_stats(json_files[:5])  # Test with first 5 files
        
        directions = [
            "North", "North-Northeast", "Northeast", "East-Northeast",
            "East", "East-Southeast", "Southeast", "South-Southeast", 
            "South", "South-Southwest", "Southwest", "West-Southwest",
            "West", "West-Northwest", "Northwest", "North-Northwest"
        ]
        
        print(f"\nWind Direction Distribution (from {total} samples):")
        print("=" * 60)
        for i, direction in enumerate(directions):
            count = direction_counts[i]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{i:2d}: {direction:18s} - {count:4d} ({percentage:5.1f}%)")
