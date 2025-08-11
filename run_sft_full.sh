#!/usr/bin/env bash
set -euo pipefail

# Full SFT run on single GPU with Unsloth
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ENABLE_UNSLOTH=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_MXFP4=1
export UNSLOTH_DISABLE_TORCH_COMPILE=1
export ACCELERATE_BYPASS_DEVICE_MAP=true
export TOKENIZERS_PARALLELISM=false
export RUN_DEMO=0
export SMOKE_MODE=0
export MODE=full
# Adjust as needed
export MAX_STEPS=${MAX_STEPS:-60}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
export SAVE_ADAPTERS=${SAVE_ADAPTERS:-1}
export ADAPTER_OUTPUT_DIR=${ADAPTER_OUTPUT_DIR:-outputs/lora_adapters}

"$(dirname "$0")/.venv-unsloth/bin/python" "$(dirname "$0")/gpt_oss_20b_fine_tuning.py"
