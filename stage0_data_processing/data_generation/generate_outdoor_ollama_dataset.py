#!/usr/bin/env python3
"""Generate outdoor literary dataset with multiple Ollama models."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Iterator

from synthetic_dataset_generator import SyntheticLiteraryDatasetGenerator

# Models to target via Ollama
OLLAMA_MODELS = {
    "mistral-small:24b",
    "llama3.1:8b",
    "gemma2:9b",
    "phi3:14b",
    "qwen2:7b",
}

OUTDOOR_SCENARIOS = [
    "city_walking",
    "forest_exploration",
    "beach_walking",
    "mountain_climbing",
    "park_stroll",
    "riverside_walking",
    "field_crossing",
    "bridge_crossing",
    "plaza_traversal",
]

WEATHER_CHOICES = ["clear", "cloudy", "fog", "windy", "rain", "snow"]
TIME_CHOICES = [
    "dawn",
    "morning",
    "forenoon",
    "noon",
    "afternoon",
    "evening",
    "night",
]

STYLE_CHOICES = [
    "modernist_novel",
    "travel_novel",
]

STYLE_TO_KEY = {
    "modernist_novel": "Modernist_Novel",
    "travel_novel": "Travel_Novel",
}


def truncated_normal(mean: float, std: float, low: float, high: float, rng: random.Random) -> float:
    while True:
        sample = rng.gauss(mean, std)
        if low <= sample <= high:
            return sample


def sample_sensor_payload(rng: random.Random) -> dict:
    temperature = round(truncated_normal(15.0, 2.0, 12.0, 25.0, rng), 1)
    humidity = round(truncated_normal(70.0, 3.0, 60.0, 75.0, rng), 1)
    scenario = rng.choice(OUTDOOR_SCENARIOS)
    weather = rng.choice(WEATHER_CHOICES)
    time_of_day = rng.choice(TIME_CHOICES)
    wind_direction_index = rng.randint(0, 15)

    heading_deg = rng.uniform(0.0, 360.0)
    heading_rad = math.radians(heading_deg)

    movement_state = "active movement" if rng.random() < 0.2 else "quiet movement"
    base_speed = 1.4 if movement_state == "active movement" else 0.35

    imu_linear_x = base_speed * math.cos(heading_rad) + rng.gauss(0.0, 0.05)
    imu_linear_y = base_speed * math.sin(heading_rad) + rng.gauss(0.0, 0.05)
    imu_linear_z = 9.8 + rng.gauss(0.0, 0.2)

    imu = [
        round(imu_linear_x, 3),
        round(imu_linear_y, 3),
        round(imu_linear_z, 3),
        round(rng.gauss(0.0, 0.12), 3),
        round(rng.gauss(0.0, 0.12), 3),
        round(rng.gauss(0.0, 0.05), 3),
    ]

    sensor = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_direction": wind_direction_index,
        "imu": imu,
        "movement_heading": round(heading_deg, 2),
        "movement_state": movement_state,
        "context": {
            "scenario": scenario,
            "time": time_of_day,
            "weather": weather,
        },
    }
    return sensor


def normalize_spacing(text: str) -> str:
    """Normalize whitespace and ensure readable spacing."""
    text = text.strip()
    if not text:
        return text

    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Ensure space after sentence-ending punctuation when needed (avoid digits for decimals)
    text = re.sub(r"([.!?;])(?![\s\"'\)\]\d])", r"\1 ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;])", r"\1", text)

    return text.strip()


def select_required_keywords(style: str, keyword_map: dict[str, list[str]], rng: random.Random) -> list[str]:
    key = STYLE_TO_KEY.get(style)
    if not key:
        return []
    words = keyword_map.get(key, [])
    if not words:
        return []
    count = 2 if len(words) < 3 else rng.randint(2, 3)
    count = min(count, len(words))
    return rng.sample(words, k=count)


def generate_records(
    generator: SyntheticLiteraryDatasetGenerator,
    model_name: str,
    count: int,
    keyword_map: dict[str, list[str]],
    seed: int | None = None,
) -> Iterator[dict]:
    rng = random.Random(seed)
    progress_step = max(1, count // 10) if count > 0 else None

    for idx in range(count):
        sensor_data = sample_sensor_payload(rng)
        style = rng.choice(STYLE_CHOICES)
        required_keywords = select_required_keywords(style, keyword_map, rng)
        generation_prompt = generator.create_literary_prompt(
            sensor_data,
            style,
            required_keywords=required_keywords,
        )
        inference_prompt = generator.create_literary_prompt(
            sensor_data,
            style,
            required_keywords=None,
        )
        try:
            response = generator._call_ollama(generation_prompt, temperature=0.7)
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed for {model_name}: {exc}")

        paragraph = normalize_spacing(response.strip())

        record_id = f"{model_name.replace(':', '_').replace('.', '')}_{idx:06d}"
        yield {
            "id": record_id,
            "model": model_name,
            "style": style,
            "sensor_data": sensor_data,
            "prompt": inference_prompt,
            "target_paragraph": paragraph,
            "metadata": {
                "required_keywords": required_keywords,
                "generation_prompt": generation_prompt,
            },
        }

        if progress_step and ((idx + 1) % progress_step == 0 or idx + 1 == count):
            print(
                f"[progress] {model_name}: {idx + 1}/{count} samples generated",
                flush=True,
            )


def write_records(records: Iterator[dict], output_path: Path, flush_every: int = 100) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[dict] = []

    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.write("[\n")
        first = True
        for record in records:
            buffer.append(record)
            if len(buffer) >= flush_every:
                for item in buffer:
                    if not first:
                        outfile.write(",\n")
                    formatted = json.dumps(item, ensure_ascii=False, indent=2)
                    indented = "\n".join(f"  {line}" for line in formatted.splitlines())
                    outfile.write(indented)
                    first = False
                buffer.clear()
        # flush remaining
        for item in buffer:
            if not first:
                outfile.write(",\n")
            formatted = json.dumps(item, ensure_ascii=False, indent=2)
            indented = "\n".join(f"  {line}" for line in formatted.splitlines())
            outfile.write(indented)
            first = False
        outfile.write("\n]\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate outdoor sensory literary dataset via multiple Ollama models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(OLLAMA_MODELS),
        required=True,
        help="Which Ollama model to use.",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=1000,
        help="Number of samples to generate (set to 100000 for full run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stage0_data_processing/generated_datasets"),
        help="Directory to save JSON datasets.",
    )
    parser.add_argument(
        "--keyword-file",
        type=Path,
        default=Path("Novel/genre_keywords-lg.json"),
        help="Path to genre keyword JSON (used to enforce required words).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed; different per model offset is applied if provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.keyword_file.exists():
        raise FileNotFoundError(f"Keyword file not found: {args.keyword_file}")

    with args.keyword_file.open("r", encoding="utf-8") as f:
        raw_keywords = json.load(f)

    keyword_map: dict[str, list[str]] = {}
    for key, words in raw_keywords.items():
        if isinstance(words, list):
            keyword_map[key] = [str(word) for word in words if isinstance(word, str)]

    model_name = args.model
    print(f"[info] Generating data for {model_name} ({args.records} samples)...")
    generator = SyntheticLiteraryDatasetGenerator(
        use_ollama=True,
        ollama_model=model_name,
    )
    generator.scenarios = OUTDOOR_SCENARIOS.copy()
    generator.literary_styles = STYLE_CHOICES.copy()
    generator.weather_contexts = WEATHER_CHOICES
    generator.time_contexts = TIME_CHOICES

    seed = args.seed
    records = generate_records(
        generator,
        model_name,
        args.records,
        keyword_map,
        seed=seed,
    )

    output_file = (
        args.output_dir
        / f"ollama_outdoor_{model_name.replace(':', '_').replace('.', '')}_{args.records}.json"
    )
    write_records(records, output_file)
    print(f"[info] Saved {output_file}")
    print("[done] Generation request finished. Review output for quality.")


if __name__ == "__main__":
    main()
