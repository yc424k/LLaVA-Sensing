"""
Training script for LLaVA-Sensing with sensor data integration.
"""

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from torch.utils.data import Dataset

from llava.train.train import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    make_supervised_data_module,
    safe_save_model_for_hf_trainer,
)
from llava.train.llava_trainer import LLaVATrainer
from stage1_training.train.modules.sensor_literature_dataset import (
    SensorLiteratureDataset,
    SensorDataCollator,
)
from stage1_training.train.modules.sensor_preprocessing import SensorDataProcessor

from llava.model import *
from llava.utils import rank0_print


@dataclass
class SensorModelArguments(ModelArguments):
    """Extended model arguments with sensor encoder configuration."""
    
    use_sensor_encoder: bool = field(default=True, metadata={"help": "Whether to use environmental sensor encoder"})
    sensor_embed_dim: int = field(default=256, metadata={"help": "Sensor embedding dimension"})
    freeze_sensor_encoder: bool = field(default=False, metadata={"help": "Whether to freeze sensor encoder during training"})


@dataclass 
class SensorDataArguments(DataArguments):
    """Extended data arguments for sensor-literature training."""
    
    sensor_data_path: str = field(default=None, metadata={"help": "Path to sensor-literature dataset JSON file"})
    use_sensor_data: bool = field(default=True, metadata={"help": "Whether to use sensor data during training"})
    data_version: str = field(default="plain", metadata={"help": "Data version"})


def make_sensor_data_module(tokenizer: transformers.PreTrainedTokenizer,
                          data_args: SensorDataArguments) -> Dict:
    """
    Create data module for sensor-literature training.
    
    Args:
        tokenizer: Tokenizer for text processing
        data_args: Data arguments including sensor data path
        
    Returns:
        Dictionary containing train dataset and data collator
    """
    
    # Create sensor data processor
    sensor_processor = SensorDataProcessor()
    
    # Create sensor-literature dataset
    train_dataset = SensorLiteratureDataset(
        data_path=data_args.sensor_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        sensor_processor=sensor_processor
    )
    
    # Create custom data collator
    data_collator = SensorDataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )


class SensorLLaVATrainer(LLaVATrainer):
    """
    Extended LLaVA trainer that handles sensor data.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with sensor data integration.
        
        Args:
            model: LLaVA model with sensor encoder
            inputs: Batch inputs including sensor data
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor (and optionally outputs)
        """
        
        # Extract sensor data if present
        sensor_data = inputs.pop('sensor_data', None)
        
        # Standard forward pass for text generation
        if sensor_data is not None:
            # Pass sensor data to model
            inputs['sensor_data'] = sensor_data
        
        return super().compute_loss(model, inputs, return_outputs)


def train():
    """Main training function for sensor-literature model."""
    
    parser = transformers.HfArgumentParser((SensorModelArguments, SensorDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_sensor_encoder:
        rank0_print("Training LLaVA-Sensing with environmental sensor encoder")
    else:
        rank0_print("Training standard LLaVA model")

    # Model setup
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_sensor_encoder = model_args.use_sensor_encoder
    config.sensor_embed_dim = model_args.sensor_embed_dim
    
    # Load model based on model type
    model_name = model_args.model_name_or_path.lower()
    if "qwen" in model_name:
        if "moe" in model_name:
            model = LlavaQwenMoeForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                trust_remote_code=True
            )
        else:
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                trust_remote_code=True
            )
    elif "mixtral" in model_name:
        model = LlavaMixtralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    elif "mistral" in model_name:
        model = LlavaMistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    elif "gemma" in model_name:
        model = LlavaGemmaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    else:
        # Default to LLaMA
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    
    # Setup special tokens
    tokenizer.pad_token = tokenizer.unk_token
    
    # Data module setup
    if data_args.use_sensor_data and data_args.sensor_data_path:
        rank0_print(f"Loading sensor-literature data from {data_args.sensor_data_path}")
        data_module = make_sensor_data_module(tokenizer=tokenizer, data_args=data_args)
    else:
        rank0_print("Using standard LLaVA data loading")
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Trainer setup
    if model_args.use_sensor_encoder:
        trainer = SensorLLaVATrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )
    else:
        trainer = LLaVATrainer(
            model=model,
            tokenizer=tokenizer, 
            args=training_args,
            **data_module
        )
    
    # Freeze vision parameters to preserve existing image capabilities (Option A)
    rank0_print("Freezing vision tower and mm_projector to preserve image capabilities")
    
    # Freeze vision tower (image encoder)
    if hasattr(model, 'vision_tower') and model.vision_tower is not None:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        rank0_print("Vision tower parameters frozen")
    
    # Freeze mm_projector (image-to-language projector)  
    if hasattr(model, 'mm_projector') and model.mm_projector is not None:
        for param in model.mm_projector.parameters():
            param.requires_grad = False
        rank0_print("MM projector parameters frozen")
    
    # Freeze most of the language model, only allow sensor-related layers to train
    if hasattr(model, 'language_model') or hasattr(model, 'model'):
        lang_model = getattr(model, 'language_model', getattr(model, 'model', None))
        if lang_model is not None:
            # Freeze embedding layers
            if hasattr(lang_model, 'embed_tokens'):
                for param in lang_model.embed_tokens.parameters():
                    param.requires_grad = False
            
            # Freeze most transformer layers (keep last few layers trainable for sensor integration)
            if hasattr(lang_model, 'layers'):
                total_layers = len(lang_model.layers)
                # Only train last 2 layers for sensor integration
                for i, layer in enumerate(lang_model.layers):
                    if i < total_layers - 2:  # Freeze all but last 2 layers
                        for param in layer.parameters():
                            param.requires_grad = False
                
                rank0_print(f"Frozen {total_layers - 2} language model layers, training last 2 layers")
    
    # Keep sensor encoder trainable (this is what we want to learn)
    if hasattr(model, 'sensor_encoder') and model.sensor_encoder is not None:
        for param in model.sensor_encoder.parameters():
            param.requires_grad = True
        rank0_print("Sensor encoder kept trainable")
    
    # Print trainable parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    rank0_print(f"Parameter Statistics:")
    rank0_print(f"  Total parameters: {total_params:,}")
    rank0_print(f"  Trainable parameters: {trainable_params:,}")
    rank0_print(f"  Frozen parameters: {frozen_params:,}")
    
    if total_params > 0:
        rank0_print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")
        rank0_print(f"  Frozen ratio: {100*frozen_params/total_params:.2f}%")
    else:
        rank0_print("  Warning: No parameters found in model")
    
    # Freeze sensor encoder if explicitly requested (override)
    if model_args.freeze_sensor_encoder and hasattr(model, 'sensor_encoder'):
        rank0_print("Overriding: Freezing sensor encoder parameters as requested")
        for param in model.sensor_encoder.parameters():
            param.requires_grad = False
    
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    # Save model
    trainer.save_state()
    
    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
