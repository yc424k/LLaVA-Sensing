"""
Complete data generation pipeline for sensor-to-literature model training.
"""

import json
import os
import asyncio
import aiohttp
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from synthetic_dataset_generator import SyntheticLiteraryDatasetGenerator, DatasetAugmentor
from prompt_templates import create_advanced_prompt, QUALITY_CHECK_PROMPT


class DatasetQualityController:
    """
    Control dataset quality through filtering and validation.
    """
    
    def __init__(self):
        self.min_length = 100
        self.max_length = 400
        self.required_keywords = ["wind", "air", "feel", "sense", "sensation", "atmosphere"]
        
    def validate_paragraph(self, paragraph: str, sensor_data: Dict) -> Dict:
        """
        Validate generated paragraph quality.
        
        Args:
            paragraph: Generated text
            sensor_data: Source sensor data
            
        Returns:
            dict: Validation results and metrics
        """
        metrics = {
            "length_valid": self.min_length <= len(paragraph) <= self.max_length,
            "has_sensory_words": any(keyword in paragraph for keyword in self.required_keywords),
            "temperature_mentioned": self._check_temperature_mention(paragraph, sensor_data["temperature"]),
            "wind_mentioned": self._check_wind_mention(paragraph),
            "coherence_score": self._calculate_coherence(paragraph),
            "repetition_score": self._calculate_repetition(paragraph)
        }
        
        # Overall quality score
        metrics["overall_score"] = sum([
            metrics["length_valid"],
            metrics["has_sensory_words"], 
            metrics["temperature_mentioned"],
            metrics["wind_mentioned"],
            metrics["coherence_score"] > 0.7,
            metrics["repetition_score"] < 0.3
        ]) / 6.0
        
        return metrics
    
    def _check_temperature_mention(self, paragraph: str, temperature: float) -> bool:
        """Check if temperature context is reflected in text."""
        if temperature < 10:
            return any(word in paragraph.lower() for word in ["cold", "cool", "chilly", "freezing", "icy", "frigid"])
        elif temperature > 25:
            return any(word in paragraph.lower() for word in ["warm", "hot", "heated", "sweltering", "scorching", "burning"])
        else:
            return True  # Moderate temperature - less strict
    
    def _check_wind_mention(self, paragraph: str) -> bool:
        """Check if wind is mentioned or implied."""
        wind_words = ["wind", "breeze", "gust", "blow", "brush", "sweep", "swirl", "whip", "rustle", "flutter"]
        return any(word in paragraph.lower() for word in wind_words)
    
    def _calculate_coherence(self, paragraph: str) -> float:
        """Calculate text coherence (simplified)."""
        sentences = paragraph.split('.')
        if len(sentences) < 2:
            return 0.8
        
        # Simple coherence check - proper sentence structure
        valid_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        return valid_sentences / len(sentences)
    
    def _calculate_repetition(self, paragraph: str) -> float:
        """Calculate repetition ratio."""
        words = paragraph.split()
        if len(words) < 5:
            return 0.0
        
        unique_words = len(set(words))
        return 1.0 - (unique_words / len(words))


class LLMInterface:
    """
    Interface for various LLM APIs.
    """
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        
    async def generate_text(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate text using configured LLM."""
        if self.provider == "openai" and self.api_key:
            return await self._openai_generate(prompt, max_tokens)
        elif self.provider == "claude":
            return await self._claude_generate(prompt, max_tokens)
        else:
            # Fallback to template generation
            return self._template_generate(prompt)
    
    async def _openai_generate(self, prompt: str, max_tokens: int) -> str:
        """OpenAI API generation."""
        import openai
        openai.api_key = self.api_key
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 뛰어난 한국어 문학 작가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return self._template_generate(prompt)
    
    async def _claude_generate(self, prompt: str, max_tokens: int) -> str:
        """Claude API generation."""
        # Implement Claude API call
        # This would require anthropic library
        return self._template_generate(prompt)
    
    def _template_generate(self, prompt: str) -> str:
        """Fallback template generation."""
        templates = [
            "공기가 차갑게 느껴지는 가운데, 바람이 얼굴을 스치며 지나갔다. 그의 발걸음은 조심스럽게 땅을 딛으며 앞으로 향했다.",
            "바람이 부드럽게 불어오자 서늘함이 느껴졌다. 촉촉한 공기 속에서 습한 기운이 감돌았고, 그는 천천히 걸어갔다.",
            "이른 아침 바람이 옷깃을 흔들었다. 차가운 공기가 뺨을 감싸는 가운데, 느린 발걸음이 딱딱한 땅 위를 울렸다."
        ]
        return np.random.choice(templates)


class BatchDatasetGenerator:
    """
    Generate large-scale dataset with quality control and parallel processing.
    """
    
    def __init__(self, 
                 llm_provider: str = "openai",
                 api_key: Optional[str] = None,
                 batch_size: int = 50,
                 max_workers: int = 4):
        
        self.generator = SyntheticLiteraryDatasetGenerator(api_key)
        self.augmentor = DatasetAugmentor()
        self.quality_controller = DatasetQualityController()
        self.llm = LLMInterface(llm_provider, api_key)
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def generate_single_example(self, 
                                    scenario: str, 
                                    time_ctx: str, 
                                    weather: str, 
                                    style: str) -> Optional[Dict]:
        """Generate single high-quality example."""
        
        # Generate sensor data
        sensor_data = self.generator.generate_realistic_sensor_data(scenario, time_ctx, weather)
        
        # Create advanced prompt
        prompt = create_advanced_prompt(sensor_data, style)
        
        # Generate paragraph
        paragraph = await self.llm.generate_text(prompt)
        
        # Validate quality
        quality_metrics = self.quality_controller.validate_paragraph(paragraph, sensor_data)
        
        # Accept only high-quality examples
        if quality_metrics["overall_score"] < 0.6:
            return None
        
        return {
            "id": f"{scenario}_{time_ctx}_{weather}_{style}_{datetime.now().strftime('%H%M%S')}",
            "sensor_data": sensor_data,
            "literary_style": style,
            "prompt": prompt,
            "target_paragraph": paragraph,
            "quality_metrics": quality_metrics,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "llm_provider": self.llm.provider,
                "context": sensor_data["context"]
            }
        }
    
    async def generate_batch(self, target_size: int) -> List[Dict]:
        """Generate batch of examples with quality control."""
        
        scenarios = ["도시_산책", "숲속_탐험", "해변_걷기", "산길_등반", "공원_산책"]
        times = ["새벽", "아침", "오후", "저녁", "밤"]
        weathers = ["맑음", "흐림", "비", "바람", "안개"]
        styles = ["모더니즘_소설", "여행기_수필", "감각적_묘사", "의식의_흐름", "자연주의_문체"]
        
        tasks = []
        attempts = 0
        max_attempts = target_size * 3  # Allow multiple attempts for quality
        
        while len(tasks) < target_size and attempts < max_attempts:
            scenario = np.random.choice(scenarios)
            time_ctx = np.random.choice(times)
            weather = np.random.choice(weathers)
            style = np.random.choice(styles)
            
            task = self.generate_single_example(scenario, time_ctx, weather, style)
            tasks.append(task)
            attempts += 1
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        self.logger.info(f"Generated {len(valid_results)} valid examples from {len(tasks)} attempts")
        
        return valid_results
    
    def augment_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Apply data augmentation to increase dataset size."""
        augmented_dataset = []
        
        for example in dataset:
            # Original example
            augmented_dataset.append(example)
            
            # Generate sensor variations
            sensor_variants = self.augmentor.augment_sensor_data(example["sensor_data"])
            
            for variant in sensor_variants[1:]:  # Skip original
                augmented_example = example.copy()
                augmented_example["id"] = f"{example['id']}_aug_{len(augmented_dataset)}"
                augmented_example["sensor_data"] = variant
                augmented_example["metadata"]["augmented"] = True
                
                augmented_dataset.append(augmented_example)
        
        return augmented_dataset
    
    def save_dataset_with_splits(self, 
                               dataset: List[Dict], 
                               output_dir: str,
                               train_ratio: float = 0.8,
                               val_ratio: float = 0.1):
        """Save dataset with train/validation/test splits."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Shuffle dataset
        np.random.shuffle(dataset)
        
        # Calculate split indices
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Create splits
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]
        
        # Save splits
        splits = {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
        
        for split_name, split_data in splits.items():
            filename = os.path.join(output_dir, f"{split_name}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved {len(split_data)} examples to {filename}")
        
        # Save metadata
        metadata = {
            "total_examples": total_size,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "generated_at": datetime.now().isoformat(),
            "generation_config": {
                "batch_size": self.batch_size,
                "quality_threshold": 0.6,
                "augmentation_applied": True
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    async def generate_full_dataset(self, 
                                  target_size: int = 1000,
                                  output_dir: str = "data/literary_dataset",
                                  apply_augmentation: bool = True) -> Dict:
        """Generate complete dataset with all processing steps."""
        
        self.logger.info(f"Starting dataset generation: target size {target_size}")
        
        # Generate base dataset
        base_dataset = await self.generate_batch(target_size // 2 if apply_augmentation else target_size)
        
        # Apply augmentation if requested
        if apply_augmentation:
            self.logger.info("Applying data augmentation...")
            final_dataset = self.augment_dataset(base_dataset)
        else:
            final_dataset = base_dataset
        
        # Trim to target size if exceeded
        if len(final_dataset) > target_size:
            final_dataset = final_dataset[:target_size]
        
        # Save with splits
        self.save_dataset_with_splits(final_dataset, output_dir)
        
        # Generate quality report
        quality_report = self.generate_quality_report(final_dataset)
        
        with open(os.path.join(output_dir, "quality_report.json"), 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        self.logger.info(f"Dataset generation complete: {len(final_dataset)} examples")
        
        return {
            "dataset_size": len(final_dataset),
            "output_directory": output_dir,
            "quality_report": quality_report
        }
    
    def generate_quality_report(self, dataset: List[Dict]) -> Dict:
        """Generate comprehensive quality report."""
        
        quality_scores = [ex.get("quality_metrics", {}).get("overall_score", 0) for ex in dataset]
        
        # Style distribution
        style_counts = {}
        for ex in dataset:
            style = ex.get("literary_style", "unknown")
            style_counts[style] = style_counts.get(style, 0) + 1
        
        # Scenario distribution
        scenario_counts = {}
        for ex in dataset:
            scenario = ex.get("sensor_data", {}).get("context", {}).get("scenario", "unknown")
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        return {
            "total_examples": len(dataset),
            "average_quality_score": np.mean(quality_scores) if quality_scores else 0,
            "quality_score_std": np.std(quality_scores) if quality_scores else 0,
            "high_quality_ratio": sum(1 for s in quality_scores if s > 0.8) / len(quality_scores) if quality_scores else 0,
            "style_distribution": style_counts,
            "scenario_distribution": scenario_counts,
            "average_paragraph_length": np.mean([len(ex.get("target_paragraph", "")) for ex in dataset])
        }


async def main():
    """Example usage of the complete pipeline."""
    
    # Initialize generator
    generator = BatchDatasetGenerator(
        llm_provider="openai",  # Change to your preferred provider
        api_key=None,  # Add your API key
        batch_size=100,
        max_workers=4
    )
    
    # Generate dataset
    result = await generator.generate_full_dataset(
        target_size=200,  # Start small for testing
        output_dir="data/sensor_literary_dataset",
        apply_augmentation=True
    )
    
    print(f"Dataset generation completed:")
    print(f"- Size: {result['dataset_size']} examples")
    print(f"- Output: {result['output_directory']}")
    print(f"- Average quality: {result['quality_report']['average_quality_score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())