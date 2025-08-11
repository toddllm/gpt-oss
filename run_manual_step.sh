#!/usr/bin/env bash
set -euo pipefail

# Manual single-step training to isolate Trainer/Accelerate overhead
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
export MANUAL_STEP=1
export MAX_SEQ_LENGTH=1024
export ATTN_IMPL=eager

"$(dirname "$0")/.venv-unsloth/bin/python" "$(dirname "$0")/gpt_oss_20b_fine_tuning.py"
