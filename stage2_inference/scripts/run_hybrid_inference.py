#!/usr/bin/env python
"""Example script to run hybrid inference with both image and sensor data."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
import requests
from io import BytesIO

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from stage1_training.train.modules.sensor_preprocessing import SensorDataProcessor
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from transformers import TextStreamer

def load_image(image_file: str) -> Image.Image:
    """Load an image from a file path or URL."""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_sensor_record(sensor_path: Path) -> Dict[str, Any]:
    """Load the first record from a sensor-literature JSON dataset."""
    with sensor_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"No entries found in {sensor_path}")
    return data[0]["sensor_data"]

def prepare_sensor_batch(sensor_record: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Preprocess sensor data and move it to the specified device."""
    processor = SensorDataProcessor()
    processed = processor.process_sensor_data(sensor_record)
    return {k: v.to(device) for k, v in processed.items()}

def main():
    parser = argparse.ArgumentParser(description="Run hybrid inference with LLaVA-Sensing.")
    parser.add_argument("--checkpoint", required=True, help="Path to the fine-tuned checkpoint directory")
    parser.add_argument("--image-file", type=str, required=True, help="Path to the image file or URL")
    parser.add_argument("--sensor_json", type=Path, default=None, help="Optional path to sensor-literature JSON file")
    parser.add_argument("--prompt", default="Generate a short literary paragraph based on the image and current environment.", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, tokenizer, and processors
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.checkpoint,
        model_name="llava_llama",
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        attn_implementation="eager",
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Prepare image data
    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device)
    if model.config.mm_projector_type == 'mlp2x_gelu':
         image_tensor = image_tensor.half()

    # Prepare sensor data
    if args.sensor_json:
        sensor_record = load_sensor_record(args.sensor_json)
    else:
        # Default sensor data if no file is provided
        sensor_record = {
            "temperature": 22.0,
            "humidity": 60.0,
            "wind_direction": 3.933,
            "imu": [0.0, -0.031, 9.777, -0.053, 0.032, 0.018],
        }
    sensor_batch = prepare_sensor_batch(sensor_record, device)

    # Build conversation prompt
    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles

    # Add image token to the prompt
    inp = args.prompt
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

    conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Setup stopping criteria and streamer
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generate text
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            sensor_data=sensor_batch,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # To get the full output text after streaming
    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    # print("=== Full Generation Output ===")
    # print(outputs)


if __name__ == "__main__":
    main()
