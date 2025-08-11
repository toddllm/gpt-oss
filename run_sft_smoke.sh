#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke test: single step, single GPU, smallest batch, no demo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ENABLE_UNSLOTH=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_MXFP4=1
export UNSLOTH_DISABLE_TORCH_COMPILE=1
export ACCELERATE_BYPASS_DEVICE_MAP=true
export RUN_DEMO=0
export MAX_STEPS=1
export SMOKE_MODE=1
# Prefer eager attention to avoid kernel init overhead in smoke
export ATTN_IMPL=eager
# Shrink context for smoke
export MAX_SEQ_LENGTH=1024

# Optional: reduce memory further for smoke
export UNSLOTH_FORCE_FLOAT32=0

"$(dirname "$0")/.venv-unsloth/bin/python" "$(dirname "$0")/gpt_oss_20b_fine_tuning.py"
