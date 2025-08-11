#!/usr/bin/env bash
set -euo pipefail

# Manual N-step training loop fallback (no TRL/Accelerate), single GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ENABLE_UNSLOTH=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_MXFP4=1
export UNSLOTH_DISABLE_TORCH_COMPILE=1
export ACCELERATE_BYPASS_DEVICE_MAP=true
export TOKENIZERS_PARALLELISM=false
export RUN_DEMO=0
export SMOKE_MODE=1
export MANUAL_LOOP=1
# Tunables
export MANUAL_STEPS=${MANUAL_STEPS:-5}
export MANUAL_BSZ=${MANUAL_BSZ:-1}
export MANUAL_GA=${MANUAL_GA:-1}
export MANUAL_LR=${MANUAL_LR:-2e-4}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-1024}
export ATTN_IMPL=eager
export SAVE_ADAPTERS=${SAVE_ADAPTERS:-1}
export ADAPTER_OUTPUT_DIR=${ADAPTER_OUTPUT_DIR:-outputs/lora_adapters}

"$(dirname "$0")/.venv-unsloth/bin/python" "$(dirname "$0")/gpt_oss_20b_fine_tuning.py"
