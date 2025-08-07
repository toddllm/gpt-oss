#!/usr/bin/env python3
"""
Run GPT-OSS-20B after model download completes
This script will work once the background download finishes
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable MXFP4 warnings
os.environ['TRANSFORMERS_NO_MXFP4'] = '1'

print("=" * 60)
print("GPT-OSS-20B Inference")
print("=" * 60)

# Check if model is downloaded
model_dir = os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b")
if not os.path.exists(model_dir):
    print("❌ Model not found. Please run ./download_model_background.sh first")
    exit(1)

# Check directory size to ensure download is complete
import subprocess
result = subprocess.run(['du', '-sb', model_dir], capture_output=True, text=True)
size_bytes = int(result.stdout.split()[0])
size_gb = size_bytes / (1024**3)

print(f"Model cache size: {size_gb:.2f} GB")
if size_gb < 35:  # Model should be ~40GB
    print(f"⚠️  Model may still be downloading (expected ~40GB, current {size_gb:.2f}GB)")
    print("Check progress with: ./monitor_download.sh")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(0)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {memory_gb:.2f}GB")

model_id = "openai/gpt-oss-20b"

print("\n" + "=" * 60)
print("Loading model...")
print("=" * 60)

try:
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with automatic settings
    print("Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for RTX 3090
        trust_remote_code=True,
        local_files_only=True,  # Use downloaded files
    )
    
    print("✅ Model loaded successfully!")
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory used: {allocated_gb:.2f} GB")
    
    # Interactive generation loop
    print("\n" + "=" * 60)
    print("Interactive GPT-OSS-20B")
    print("Type 'quit' to exit, 'clear' to clear screen")
    print("=" * 60)
    
    while True:
        prompt = input("\n📝 Enter prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'clear':
            os.system('clear')
            continue
        elif not prompt:
            continue
        
        print("\n🤖 Generating...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and print
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Only show the generated part
        generated = response[len(prompt):].strip()
        print(f"\n💬 Response: {generated}")
    
    print("\n👋 Goodbye!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nTroubleshooting:")
    print("1. Ensure download is complete: ./monitor_download.sh")
    print("2. Check GPU memory: nvidia-smi")
    print("3. Try clearing GPU memory and running again")
    print("4. If OOM, try with load_in_8bit=True")