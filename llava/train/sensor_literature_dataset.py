"""
Custom dataset class for sensor-literature training.
"""

import json
import torch
import random
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import transformers

from llava.train.sensor_preprocessing import SensorDataProcessor, convert_to_llava_format, load_sensor_literature_data
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from llava.train.train import preprocess_multimodal, preprocess_llama_2, preprocess_v1, preprocess_plain
from llava import conversation as conversation_lib


class SensorLiteratureDataset(Dataset):
    """
    Dataset class for training LLaVA-Sensing with sensor data and literary text.
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 sensor_processor: Optional[SensorDataProcessor] = None):
        """
        Initialize sensor-literature dataset.
        
        Args:
            data_path: Path to sensor-literature JSON dataset
            tokenizer: Tokenizer for text processing
            data_args: Data arguments from training script
            sensor_processor: Sensor data processor (creates default if None)
        """
        super(SensorLiteratureDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        # Initialize sensor processor
        if sensor_processor is None:
            self.sensor_processor = SensorDataProcessor()
        else:
            self.sensor_processor = sensor_processor
        
        # Load and convert data
        print(f"Loading sensor-literature data from {data_path}")
        raw_data = load_sensor_literature_data(data_path)
        self.list_data_dict = convert_to_llava_format(raw_data)
        
        print(f"Loaded {len(self.list_data_dict)} sensor-literature pairs")
        
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            i: Index of the example
            
        Returns:
            Dictionary containing processed data for training
        """
        # Data is already in LLaVA format from convert_to_llava_format
        data_item = self.list_data_dict[i]
        
        # Extract conversation from data item and format for preprocessing
        conversations = data_item["conversations"]
        sources = [conversations]  # preprocess_qwen expects list of conversation lists
        
        # Process conversation text for sensor-only training (no images)
        # Use qwen preprocessing for LLaVA-OneVision Qwen2 model
        from llava.train.train import preprocess_qwen
        sources = preprocess_qwen(sources, self.tokenizer, has_image=False)
        
        # Handle different return types from preprocess_qwen
        if isinstance(sources, dict):
            # preprocess_qwen returned a dict directly
            if 'input_ids' in sources and 'labels' in sources:
                data_dict = dict(
                    input_ids=sources['input_ids'],
                    labels=sources['labels']
                )
            else:
                print(f"Error - Dict sources missing required keys: {sources.keys()}")
                raise ValueError("Dict sources missing input_ids or labels")
        elif isinstance(sources, list) and len(sources) > 0:
            # preprocess_qwen returned a list (original expected behavior)
            if isinstance(sources[0], dict):
                data_dict = dict(
                    input_ids=sources[0]['input_ids'],
                    labels=sources[0]['labels']
                )
            else:
                data_dict = dict(
                    input_ids=sources[0],
                    labels=sources[0].clone()
                )
        else:
            # Handle empty or invalid sources
            print(f"Error - Invalid sources returned: {type(sources)}, {sources}")
            raise ValueError("Invalid sources returned from preprocess_qwen")
        
        # Process sensor data
        sensor_data = self.list_data_dict[i]['sensor_data']
        processed_sensors = self.sensor_processor.process_sensor_data(sensor_data)
        
        # Add processed sensor data to return dictionary
        data_dict.update({
            'sensor_temperature': processed_sensors['temperature'],
            'sensor_humidity': processed_sensors['humidity'], 
            'sensor_wind_direction': processed_sensors['wind_direction'],
            'sensor_imu': processed_sensors['imu'],
            'has_sensor_data': torch.tensor(True)
        })
        
        return data_dict


class SensorDataCollator:
    """
    Data collator that handles both text and sensor data batching.
    """
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, pad_token_id: int = None):
        """
        Initialize data collator.
        
        Args:
            tokenizer: Tokenizer for text processing
            pad_token_id: Token ID for padding (uses tokenizer.pad_token_id if None)
        """
        self.tokenizer = tokenizer
        # Ensure we have a valid pad_token_id
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            self.pad_token_id = tokenizer.eos_token_id
            print(f"Warning: Using eos_token_id ({self.pad_token_id}) as pad_token_id")
        else:
            self.pad_token_id = 0
            print(f"Warning: No pad_token_id found, using 0 as fallback")
        
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances.
        
        Args:
            instances: List of data instances from dataset
            
        Returns:
            Batched data ready for training
        """
        batch_size = len(instances)
        
        # Tokenize and pad text sequences
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        
        # Debug: Print shapes and ensure they match
        print(f"Debug: Batch size = {len(input_ids)}")
        for i, (inp_ids, lbls) in enumerate(zip(input_ids, labels)):
            print(f"Instance {i}: input_ids shape={inp_ids.shape}, labels shape={lbls.shape}")
            
            # Ensure both are 1D tensors
            if len(inp_ids.shape) != 1:
                inp_ids = inp_ids.flatten()
                input_ids[i] = inp_ids
                print(f"  Fixed input_ids to 1D: {inp_ids.shape}")
            
            if len(lbls.shape) != 1:
                lbls = lbls.flatten() 
                labels[i] = lbls
                print(f"  Fixed labels to 1D: {lbls.shape}")
            
            # Ensure same length
            if inp_ids.shape[0] != lbls.shape[0]:
                print(f"  Shape mismatch: input_ids={inp_ids.shape[0]}, labels={lbls.shape[0]}")
                min_len = min(inp_ids.shape[0], lbls.shape[0])
                input_ids[i] = inp_ids[:min_len]
                labels[i] = lbls[:min_len]
                print(f"  Fixed: Truncated both to length {min_len}")
        
        try:
            # Pad sequences
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
        except Exception as e:
            print(f"Error during padding: {e}")
            # Print final shapes for debugging
            for i, (inp_ids, lbls) in enumerate(zip(input_ids, labels)):
                print(f"Final instance {i}: input_ids={inp_ids.shape}, labels={lbls.shape}")
            raise
        
        # Collect sensor data
        sensor_temperature = torch.cat([instance['sensor_temperature'] for instance in instances], dim=0)
        sensor_humidity = torch.cat([instance['sensor_humidity'] for instance in instances], dim=0)
        sensor_wind = torch.cat([instance['sensor_wind_direction'] for instance in instances], dim=0)
        sensor_imu = torch.cat([instance['sensor_imu'] for instance in instances], dim=0)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.pad_token_id),
            'sensor_data': {
                'temperature': sensor_temperature,
                'humidity': sensor_humidity,
                'wind_direction': sensor_wind,
                'imu': sensor_imu
            }
        }


def create_sensor_literature_dataset(data_path: str, 
                                   tokenizer: transformers.PreTrainedTokenizer,
                                   data_args) -> SensorLiteratureDataset:
    """
    Factory function to create sensor-literature dataset.
    
    Args:
        data_path: Path to sensor-literature data
        tokenizer: Tokenizer for text processing
        data_args: Data arguments
        
    Returns:
        Configured dataset instance
    """
    return SensorLiteratureDataset(
        data_path=data_path,
        tokenizer=tokenizer, 
        data_args=data_args
    )