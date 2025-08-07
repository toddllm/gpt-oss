# GPT-OSS-20B Setup Guide

## Current Status
- ✅ Environment configured (PyTorch 2.6.0, Triton 3.2.0)
- ✅ Download started in background
- ⏳ Waiting for ~40GB model download to complete

## Monitor Download Progress

```bash
# Check current status
./monitor_download.sh

# Watch live progress
tail -f download_progress.log

# Check download size
du -sh ~/.cache/huggingface/hub/models--openai--gpt-oss-20b
```

## After Download Completes

Run the model with:
```bash
python run_after_download.py
```

This will:
1. Check if download is complete
2. Load the model with optimal settings for RTX 3090
3. Provide an interactive chat interface

## Alternative: vLLM (Better Performance)

Once download completes, you can also use vLLM:
```bash
python test_vllm.py
```

## Files Created

- `download_model_background.sh` - Downloads model in background
- `monitor_download.sh` - Check download progress
- `run_after_download.py` - Run model after download
- `test_vllm.py` - Alternative using vLLM
- `download_progress.log` - Download log file
- `download.pid` - Process ID of download

## Troubleshooting

If download stalls:
```bash
# Check if process is still running
ps -p $(cat download.pid)

# Kill and restart if needed
kill $(cat download.pid)
./download_model_background.sh
```

If model won't load after download:
- RTX 3090 has 24GB VRAM, enough for 20B model in bfloat16
- If OOM, try reducing batch size or sequence length
- Clear GPU memory: `nvidia-smi` then kill any Python processes

## Technical Notes

- **MXFP4**: Not supported on RTX 3090 (Ampere). Requires Hopper/Blackwell GPUs.
- **Quantization**: Using bfloat16 by default, which fits in 24GB VRAM
- **Triton**: Using 3.2.0 to match PyTorch 2.6.0 requirements