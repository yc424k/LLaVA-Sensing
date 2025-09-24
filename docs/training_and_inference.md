# LLaVA-Sensing Training & Inference Guide

This document summarizes how to fine-tune and run inference with the sensor-augmented LLaVA model in this repository.

---

## 0. Activate Environment

```bash
conda activate SensingLLaVA
cd /home/yc424k/LLaVA-Sensing
```

Ensure PyTorch and project dependencies are installed in this environment.

---

## 1. Prepare Sensor JSON Dataset

The training scripts expect a single JSON file where each entry contains `sensor_data` and a `target_paragraph`. If your data is split into multiple chunk folders (e.g., `modernist_test` and `travel_test`), merge them first:

```bash
python stage0_data_processing/scripts/data/merge_sensor_chunks.py \
  stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_test \
  stage0_data_processing/data_generation/data/processed/test_val_30k_each/travel_test \
  --output stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_travel_test.json
```

### Sensor entry preview

```json
{
  "id": "A167_A_Son_at_the_Front_hybrid_085",
  "sensor_data": {
    "temperature": 15.0,
    "humidity": 60.0,
    "wind_direction": 14,
    "imu": [-0.405, 0.431, 9.95, -0.046, 0.013, -0.041],
    "context": {"scenario": "city_walking", "time": "afternoon", "weather": "clear"}
  },
  "target_paragraph": "Campton had fled to Montmartre ...",
  "metadata": {...}
}
```

`SensorDataProcessor` will normalize these values for the model.

---

## 2. Fine-tuning

Set environment variables and launch the training script.

```bash
export SENSOR_DATA_PATH="stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_travel_test.json"
export OUTPUT_DIR="checkpoints/sensor-literature/llama3-8b-modernist-travel-$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true          # optional
# Optional extra trainer args (e.g., limit steps)
export EXTRA_ARGS="--max_steps 1000"

bash stage1_training/train/finetune_sensor_literature_llama3_8b.sh $EXTRA_ARGS
```

Key defaults in the script:
- Base model: `lmms-lab/llava-llama3-8b`
- Trainable modules: `mm_mlp_adapter`, `mm_language_model`, `sensor_encoder`
- Batch: `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
- BF16 training, ZeRO-3 via `scripts/zero3.json`

### Monitoring progress

* Check live progress (reads `trainer_state.json`):
  ```bash
  watch -n 10 "python scripts/utils/show_training_progress.py --output_dir $OUTPUT_DIR"
  ```
* Log training output to file:
  ```bash
  bash stage1_training/train/finetune_sensor_literature_llama3_8b.sh $EXTRA_ARGS | tee $OUTPUT_DIR/train.log
  ```

### Resuming

To resume, pass `--resume_from_checkpoint $OUTPUT_DIR/checkpoint-XXXX` via `EXTRA_ARGS`.

---

## 3. Inference (Sensor-conditioned generation)

Use the helper script to load the fine-tuned checkpoint and generate text from sensor readings.

```bash
python stage2_inference/scripts/run_sensor_inference.py \
  --checkpoint checkpoints/sensor-literature/llama3-8b-modernist-travel-20250922 \
  --sensor_json stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_travel_test.json \
  --prompt "Describe the current environment in an evocative literary paragraph." \
  --max_new_tokens 500 \
  --temperature 0.7
```

What happens internally:
- Sensor values are normalized (`SensorDataProcessor`) → tensor batch.
- Image (optional) is preprocessed via `image_processor` if `--image_file` is provided.
- Prompt is wrapped with conversation template (`llava_v1`) and tokenized with attention mask.
- Model runs `generate(inputs=..., attention_mask=..., images=?, sensor_data=...)` to produce text.

### Notes on warnings

- FlashAttention 2 is disabled (`attn_implementation="eager"`).
- Pad token is set to EOS to avoid tokenizer warnings.
- Attention mask is now supplied, so generation is stable.

### Adjusting generation

- `--max_new_tokens`: controls maximum generated tokens.
- `--temperature`: higher → more diverse output.

### Using images as well

`--image_file` accepts a local path or URL. When supplied, the script loads the image, injects the image token into the prompt, and forwards both `images` and `sensor_data` to `model.generate`.

---

## 4. Common warnings explained

* **Meta device / set_module_tensor_to_device**: benign; occurs with accelerate’s memory-efficient loading.
* **Attention mask / pad token**: resolved by setting pad token and attention mask in the inference script.

---

## 5. Quick reference: sample python snippet to inspect sensor JSON

```python
import json
path = 'stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_travel_test.json'
with open(path, 'r', encoding='utf-8') as f:
    entry = json.load(f)[0]
print(entry['sensor_data'])
print(entry['target_paragraph'][:200], '...')
```

---

## 6. Summary of key files

| File | Purpose |
|------|---------|
| `stage1_training/train/finetune_sensor_literature_llama3_8b.sh` | Training launcher (LLaMA3 + sensor encoder) |
| `stage0_data_processing/scripts/data/merge_sensor_chunks.py` | Merge chunked sensor JSON files |
| `stage2_inference/scripts/run_sensor_inference.py` | Sensor-conditioned text generation |
| `scripts/utils/show_training_progress.py` | Monitor training progress via `trainer_state.json` |

---

All commands assume the repository root is `/home/yc424k/LLaVA-Sensing`. Adjust paths if running elsewhere.
