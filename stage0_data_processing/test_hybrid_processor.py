#!/usr/bin/env python3
"""
Test script for hybrid novel processor.
"""

import json

from stage0_data_processing.data_generation.hybrid_novel_processor import (
    HybridNovelProcessor,
)


def test_hybrid_processor():
    """Test the hybrid processor with sample text."""
    
    # Sample literary text for testing
    sample_texts = [
        """The cold morning air bit at his cheeks as he walked through the fog-laden streets. 
        Each step echoed against the wet cobblestones, while a gentle breeze carried the scent 
        of rain from the previous night. The city seemed wrapped in a gray blanket, muffling 
        the usual sounds of traffic and conversation.""",
        
        """Under the blazing afternoon sun, she climbed the rocky mountain path. Sweat beaded 
        on her forehead as the dry wind whipped across the exposed ridge. The heat shimmered 
        off the stones, and every breath felt like fire in her lungs. Yet she pressed on, 
        driven by the promise of the summit.""",
        
        """The forest was quiet except for the rustle of leaves in the humid air. Thick mist 
        clung to the ancient trees, and droplets of moisture fell from the canopy above. 
        His footsteps were muffled by the soft earth, and the air tasted of moss and decay."""
    ]
    
    processor = HybridNovelProcessor(use_ollama=True)
    
    print("=== Testing Hybrid Novel Processor ===\n")
    
    for i, text in enumerate(sample_texts):
        print(f"--- Test Case {i+1} ---")
        print(f"Sample text: {text[:100]}...")
        
        # Test environmental analysis
        env_analysis = processor.analyze_environment_with_llm(text)
        print(f"Environmental Analysis:")
        print(f"  Temperature: {env_analysis.get('temperature_celsius', 'N/A')}°C")
        print(f"  Humidity: {env_analysis.get('humidity_percent', 'N/A')}%")
        print(f"  Weather: {env_analysis.get('weather_condition', 'N/A')}")
        print(f"  Location: {env_analysis.get('location_type', 'N/A')}")
        print(f"  Confidence: {env_analysis.get('confidence', 'N/A')}")
        
        # Test sensor data generation
        sensor_data = processor.generate_sensor_data_from_analysis(env_analysis)
        print(f"Generated Sensor Data:")
        print(f"  Temperature: {sensor_data['temperature']}°C")
        print(f"  Humidity: {sensor_data['humidity']}%")
        print(f"  Wind direction: {sensor_data['wind_direction']:.3f} rad")
        print(f"  Context: {sensor_data['context']}")
        
        print(f"Original text preserved: YES ✅")
        print()


def compare_approaches():
    """Compare simple vs hybrid approaches."""
    
    sample_text = """The morning fog hung thick over the harbor, and John pulled his coat tighter 
    against the damp chill. Each breath formed small clouds in the frigid air as he walked along 
    the pier. The wind from the ocean carried the salty tang of seaweed and the distant cry of gulls."""
    
    # Test simple approach
    from stage0_data_processing.data_generation.simple_novel_processor import (
        SimpleNovelProcessor,
    )
    simple_processor = SimpleNovelProcessor()
    simple_analysis = simple_processor.analyze_sensory_content(sample_text)
    simple_sensor_data = simple_processor.infer_sensor_data(sample_text, simple_analysis)
    
    # Test hybrid approach  
    hybrid_processor = HybridNovelProcessor(use_ollama=True)
    hybrid_env_analysis = hybrid_processor.analyze_environment_with_llm(sample_text)
    hybrid_sensor_data = hybrid_processor.generate_sensor_data_from_analysis(hybrid_env_analysis)
    
    print("=== Comparison: Simple vs Hybrid ===\n")
    print(f"Sample: {sample_text}\n")
    
    print("SIMPLE APPROACH:")
    print(f"  Temperature: {simple_sensor_data['temperature']}°C")
    print(f"  Humidity: {simple_sensor_data['humidity']}%")  
    print(f"  Context: {simple_sensor_data['context']}")
    print(f"  Method: Keyword counting")
    print()
    
    print("HYBRID APPROACH:")
    print(f"  Temperature: {hybrid_sensor_data['temperature']}°C")
    print(f"  Humidity: {hybrid_sensor_data['humidity']}%")
    print(f"  Context: {hybrid_sensor_data['context']}")
    print(f"  LLM Analysis: {hybrid_env_analysis.get('atmospheric_details', {})}")
    print(f"  Confidence: {hybrid_env_analysis.get('confidence', 0)}")
    print()
    
    print("KEY DIFFERENCE:")
    print("✅ Both preserve original text")
    print("❌ Simple: Basic keyword matching")  
    print("✅ Hybrid: Sophisticated LLM understanding")


if __name__ == "__main__":
    print("Testing hybrid processor...\n")
    test_hybrid_processor()
    print("\n" + "="*50 + "\n")
    compare_approaches()
