#!/usr/bin/env python3
"""
Simple test to load and run gpt-oss-20b
"""

import torch
import gc
import os

# Set memory-efficient settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("Testing gpt-oss-20b loading...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

try:
    # Try with bitsandbytes if available
    import bitsandbytes as bnb
    print("✓ bitsandbytes available")
    use_bnb = True
except ImportError:
    print("⚠ bitsandbytes not available")
    use_bnb = False

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openai/gpt-oss-20b"

print(f"\nAttempting to load {model_id}...")

# Load tokenizer first
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Try different loading strategies
strategies = [
    ("4-bit quantization", {"load_in_4bit": True, "device_map": "auto", "torch_dtype": torch.float16}),
    ("8-bit quantization", {"load_in_8bit": True, "device_map": "auto", "torch_dtype": torch.float16}),
    ("FP16", {"torch_dtype": torch.float16, "device_map": "auto", "low_cpu_mem_usage": True}),
]

model = None
for name, kwargs in strategies:
    try:
        print(f"\nTrying {name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **kwargs
        )
        print(f"✓ Successfully loaded with {name}")
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        break
        
    except Exception as e:
        print(f"✗ Failed with {name}: {str(e)[:100]}")
        torch.cuda.empty_cache()
        gc.collect()
        continue

if model is None:
    print("\n❌ Could not load model with any strategy")
    exit(1)

# Test generation
print("\n" + "="*60)
print("Testing generation...")
print("="*60)

prompt = "The future of artificial intelligence is"
print(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResponse:\n{response}")

print("\n✓ Test successful!")