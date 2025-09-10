# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is LLaVA-Sensing, a specialized fork of LLaVA-NeXT (Large Language and Vision Assistant) that incorporates environmental sensor data for multimodal AI applications. The project extends the original LLaVA architecture to handle sensor inputs like temperature, humidity, wind direction, and IMU data alongside visual and textual information.

## Key Architecture

### Core Components

- **LLaVA Base Architecture**: Built on the LLaVA-NeXT foundation with support for images, videos, and multi-modal inputs
- **Environmental Sensor Encoder**: Custom encoder at `llava/model/multimodal_encoder/environmental_sensor_encoder.py` for processing sensor data
- **Language Model Integration**: Supports multiple LLMs (Llama, Qwen, Mistral, Mixtral, Gemma)
- **Vision Tower**: CLIP/SigLIP-based vision encoders for image understanding
- **Multimodal Fusion**: Advanced attention mechanisms to combine vision, text, and sensor modalities

### Sensor Integration

The environmental sensor encoder handles:
- Temperature and humidity sensors
- Wind direction (global and robot-relative)
- IMU data (accelerometer + gyroscope)
- Cross-modal attention for sensor fusion
- Literary interpretation for English language output

### Data Generation Pipeline

- Located in `data_generation/` directory
- Processes large datasets into chunks for training
- Custom dataset readers and processors
- Novel dataset processing with chunking capabilities

## Installation & Setup

```bash
# Create conda environment
conda create -n llava python=3.10 -y
conda activate llava

# Install dependencies
pip install --upgrade pip
pip install -e ".[train]"
```

## Common Development Commands

### Training

```bash
# Single image training
bash scripts/train/finetune_si.sh

# OneVision training
bash scripts/train/finetune_ov.sh

# DPO training
bash scripts/train/dpo.sh

# Pretraining
bash scripts/train/pretrain_clip.sh
```

### Evaluation

Use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework for evaluation across multiple benchmarks.

### Model Inference

```bash
# Basic inference using predict.py for Cog deployment
python predict.py

# Use SGLang for faster inference and deployment
# See documentation in README for SGLang setup
```

### Data Processing

```bash
# Process datasets into chunks
python data_generation/dataset_pipeline.py

# Split datasets
python data_generation/split_dataset.py

# Process novel datasets
python data_generation/simple_novel_processor.py
```

## Code Organization

### Model Architecture Files

- `llava/model/llava_arch.py` - Core multimodal architecture and fusion logic
- `llava/model/builder.py` - Model loading and configuration
- `llava/model/language_model/` - Language model implementations for different base models
- `llava/model/multimodal_encoder/` - Vision and sensor encoders
- `llava/model/multimodal_projector/` - Cross-modal projection layers

### Training Components

- `llava/train/` - Training scripts and utilities
- `scripts/train/` - Shell scripts for various training configurations
- Configuration files use YAML format (see `scripts/train/*.yaml`)

### Evaluation & Serving

- `llava/eval/` - Evaluation utilities
- `llava/serve/` - Model serving components
- `predict.py` - Cog-compatible prediction interface

## Environment Configuration

### Key Dependencies

- PyTorch 2.1.2+ with CUDA support
- Transformers (custom version from HuggingFace)
- DeepSpeed for distributed training
- Flash Attention for efficient attention computation
- OpenAI CLIP for vision processing
- Various sensor processing libraries

### Hardware Requirements

- GPU with sufficient VRAM (8GB+ recommended)
- CUDA-compatible GPU for training
- Multi-GPU support via DeepSpeed

## Project-Specific Notes

### Sensor Data Format

Sensor inputs are expected as dictionaries with keys:
- `temperature`: Float tensor [batch_size, 1] in Celsius
- `humidity`: Float tensor [batch_size, 1] in relative humidity (0-100)
- `wind_direction`: Float tensor [batch_size, 1] in radians
- `imu`: Float tensor [batch_size, 6] with [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

### Korean Language Support

The sensor encoder includes Korean literary descriptions for wind direction interpretation, suggesting this project targets Korean language applications.

### Model Variants

Supports various base models:
- LLaMA (multiple versions)
- Qwen (including MoE variants)
- Mistral/Mixtral
- Gemma
- Custom LoRA training support

### Configuration Management

- Uses Black formatter with 240 character line length
- DeepSpeed configurations available in `scripts/zero*.json`
- Model configurations via YAML files in `scripts/train/`

## Debugging & Development

### Common Issues

- Memory issues: Use DeepSpeed zero configurations
- Model loading: Check model name patterns in `builder.py`
- Sensor integration: Verify sensor data format matches encoder expectations

### Development Tips

- Use `rank0_print()` for distributed training logs
- Flash attention is enabled by default for efficiency
- Vision tower loading can be delayed with `delay_load=True`
- Multiple evaluation frameworks supported via lmms-eval

## LLaVA-Critic Integration

The repository includes LLaVA-Critic-R1 components in `llava-critic-r1/EasyR1/` for reinforcement learning from human feedback (RLHF) and generative critique capabilities.

## Final Research Plan
1. Research Objective
Develop an AI model that utilizes real-world sensor data (images, temperature/humidity, wind, IMU) collected by robots as 'inspiration' similar to human novelists, generating literary paragraphs rich in sensory descriptions.
2. Core Methodology
Learning Data
To train the model in modernist novel and travelogue styles, construct a high-quality synthetic dataset by inputting structured prompts (situational data + stylistic requirements) into large language models.
Data Processing

Utilize image, temperature/humidity, wind data along with IMU sensors to calculate 'subjective wind' relative to the robot's direction of movement
Convert these sensor data into vector format that the model can understand

AI Model
Employ a multi-modal architecture that takes each data type as separate input channels, effectively fusing different types of sensory information.
3. Research Scope and Limitations
Output
Focus on generating 'description-centered paragraphs' for specific situations, rather than complete novels with finished narratives.
Clear Limitations
This research focuses on the literary transformation of sensory data and explicitly excludes the generation of complex plots with causal relationships.