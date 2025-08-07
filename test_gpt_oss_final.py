#!/usr/bin/env python3
"""Test GPT-OSS-20B with proper configuration"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment variables
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TRANSFORMERS_NO_MXFP4'] = '1'  # Disable MXFP4 quantization

print("=" * 60)
print("GPT-OSS-20B Model Test")
print("=" * 60)

# Model location
model_path = "/home/tdeshane/gpt-oss/clean_model"

print(f"\n1. Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("   ✓ Tokenizer loaded")

print(f"\n2. Loading model (this will take a while)...")
print("   Using FP16 precision to save memory")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("   ✓ Model loaded successfully")
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params/1e9:.2f}B")
    
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    print("\n   Trying with HuggingFace hub fallback...")
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

print("\n3. Running inference test...")

# Test prompt
prompt = "The future of artificial intelligence is"
print(f"   Prompt: '{prompt}'")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
print("   Generating text...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "=" * 60)
print("Generated text:")
print(result)
print("=" * 60)

print("\n✓ Test completed successfully!")

# Memory usage
if torch.cuda.is_available():
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    print("Running on CPU")