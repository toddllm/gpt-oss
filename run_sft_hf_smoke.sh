#!/usr/bin/env bash
set -euo pipefail

# TRL 1-step smoke test using HF+bitsandbytes path (no Unsloth patches)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ENABLE_UNSLOTH=0
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_MXFP4=1
export ACCELERATE_BYPASS_DEVICE_MAP=true
export TOKENIZERS_PARALLELISM=false
export RUN_DEMO=0
export MAX_STEPS=1
export SMOKE_MODE=1
export ATTN_IMPL=eager
export MAX_SEQ_LENGTH=1024

"$(dirname "$0")/.venv-unsloth/bin/python" "$(dirname "$0")/gpt_oss_20b_fine_tuning.py"
