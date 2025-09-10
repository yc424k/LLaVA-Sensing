#!/bin/bash

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export ACCELERATE_CPU_AFFINITY=0

# Model configuration
LLM_VERSION="lmms-lab/llava-onevision-qwen2-7b-si"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"

# Sensor-Literature Training Configuration
RUN_NAME="llava-sensing-${LLM_VERSION_CLEAN}-sensor-literature-$(date +%Y%m%d)"
echo "RUN_NAME: ${RUN_NAME}"

# Paths
SENSOR_DATA_PATH="data_generation/data/processed/LLaMA 3.2 3B 4bit/novel_dataset_chunk_001.json"  # Using LLaMA 3.2 3B 4bit dataset chunks 1-100
OUTPUT_DIR="./checkpoints/sensor-literature/${RUN_NAME}"
echo "SENSOR_DATA_PATH: ${SENSOR_DATA_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

# Training parameters
NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
RANK=${RANK:-0}
ADDR=${ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}

echo "Training on ${NUM_GPUS} GPUs across ${NNODES} nodes"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_sensor_literature.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version plain \
    --sensor_data_path "${SENSOR_DATA_PATH}" \
    --use_sensor_data True \
    --use_sensor_encoder True \
    --sensor_embed_dim 256 \
    --freeze_sensor_encoder False \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model,sensor_encoder" \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --remove_unused_columns False

echo "Training completed. Model saved to ${OUTPUT_DIR}"