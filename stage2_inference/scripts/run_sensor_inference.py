#!/usr/bin/env python
"""Simple example script to run sensor-conditioned text generation with a fine-tuned LLaVA-Sensing model."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

import requests
import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.model.builder import load_pretrained_model
from stage1_training.train.modules.sensor_preprocessing import SensorDataProcessor
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


def load_sensor_record(sensor_path: Path) -> Dict[str, Any]:
    """Load the first record from a sensor-literature JSON dataset."""
    with sensor_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"No entries found in {sensor_path}")
    return data[0]["sensor_data"]


def load_image(image_path: str) -> Image.Image:
    """Load an image either from disk or a URL."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image


def prepare_sensor_batch(sensor_record: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    processor = SensorDataProcessor()
    processed = processor.process_sensor_data(sensor_record)
    return {k: v.to(device) for k, v in processed.items()}


def build_prompt(prompt: str, conv_mode: str, include_image: bool, model_config) -> str:
    # Use a conversation template that does not require gated LLaMA-3 tokenizer access.
    conv = conv_templates[conv_mode].copy()
    if include_image:
        if getattr(model_config, "mm_use_im_start_end", False):
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LLaVA-Sensing model.")
    parser.add_argument("--checkpoint", required=True, help="Path to the fine-tuned checkpoint directory")
    parser.add_argument("--prompt", default="Generate a short literary paragraph based on the current environment.", help="Prompt text")
    parser.add_argument("--sensor_json", type=Path, default=None, help="Optional path to sensor-literature JSON file")
    parser.add_argument("--image_file", type=str, default=None, help="Optional path or URL to an image to condition on")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--conv_mode", type=str, default="llava_v1", help="Conversation template to use")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    # Pre-load tokenizer to determine vocabulary size (e.g., LLaMA-3 uses 128k)
    device_map = {"": "cuda:0"} if device.type == "cuda" else {"": "cpu"}

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.checkpoint,
        model_name="llava_llama",
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        attn_implementation="eager",
        device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    image_tensor = None
    if args.image_file is not None:
        if image_processor is None:
            raise ValueError("Loaded model does not provide an image processor but --image_file was supplied.")
        image = load_image(args.image_file)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device)
        if getattr(model.config, "mm_projector_type", "") == "mlp2x_gelu":
            image_tensor = image_tensor.half()

    # Sensor data prepared from user file or default sample
    if args.sensor_json is not None:
        sensor_record = load_sensor_record(args.sensor_json)
    else:
        sensor_record = {
            "temperature": 22.0,
            "humidity": 60.0,
            "wind_direction": 3.933,
            "imu": [0.0, -0.031, 9.777, -0.053, 0.032, 0.018],
        }
    sensor_batch = prepare_sensor_batch(sensor_record, device)

    prompt = build_prompt(args.prompt, args.conv_mode, image_tensor is not None, model.config)
    if image_tensor is not None:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        attention_mask = None
    else:
        encoded = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        generate_kwargs = dict(
            inputs=input_ids,
            sensor_data=sensor_batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if image_tensor is not None:
            generate_kwargs["images"] = image_tensor

        output_ids = model.generate(**generate_kwargs)

    generated_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("=== Sensor-conditioned generation ===")
    print(text)


if __name__ == "__main__":
    main()
