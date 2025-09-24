#!/bin/bash

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extra_args)
      shift
      [[ $# -gt 0 ]] || { echo "Error: --extra_args requires an argument" >&2; exit 1; }
      read -r -a __parsed <<< "$1"
      EXTRA_ARGS+=("${__parsed[@]}")
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export ACCELERATE_CPU_AFFINITY=${ACCELERATE_CPU_AFFINITY:-0}

LLM_VERSION="lmms-lab/llama3-llava-next-8b"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"

RUN_NAME="llava-sensing-${LLM_VERSION_CLEAN}-sensor-literature-$(date +%Y%m%d)"
echo "RUN_NAME: ${RUN_NAME}"

: "${SENSOR_DATA_PATH:=stage0_data_processing/data_generation/data/processed/LLaMA 3.2 3B 4bit/novel_dataset_chunk_001.json}" >/dev/null
: "${OUTPUT_DIR:=./checkpoints/sensor-literature/${RUN_NAME}}" >/dev/null

echo "Using base model: ${LLM_VERSION}"
echo "Sensor dataset: ${SENSOR_DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching training with ${NUM_GPUS} GPU(s) across ${NNODES} node(s)"

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "/home/yc424k/.conda/envs/SensingLLaVA/bin/python" ]]; then
    PYTHON_BIN="/home/yc424k/.conda/envs/SensingLLaVA/bin/python"
  else
    PYTHON_BIN=$(command -v python)
  fi
fi

if [[ "${NUM_GPUS}" -gt 1 || "${NNODES}" -gt 1 ]]; then
  LAUNCH_CMD=("${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NNODES}" \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}")
else
  LAUNCH_CMD=("${PYTHON_BIN}")
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${LAUNCH_CMD[@]}" stage1_training/train/modules/train_sensor_literature.py \
  ${DEEPSPEED_CONFIG:+--deepspeed ${DEEPSPEED_CONFIG}} \
  --model_name_or_path "${LLM_VERSION}" \
  --version plain \
  --sensor_data_path "${SENSOR_DATA_PATH}" \
  --use_sensor_data True \
  --use_sensor_encoder True \
  --sensor_embed_dim 256 \
  --freeze_sensor_encoder False \
  --mm_tunable_parts "mm_mlp_adapter,mm_language_model,sensor_encoder" \
  --bf16 True \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 32768 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --remove_unused_columns False \
  "${EXTRA_ARGS[@]}"

echo "Training finished. Checkpoints written to ${OUTPUT_DIR}"
