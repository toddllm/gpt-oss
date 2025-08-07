#!/usr/bin/env python3
"""Simple test of GPT-OSS-20B model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading GPT-OSS-20B model...")
print("This will use the Hugging Face cache system")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
print("✓ Tokenizer loaded")

# Load model with reduced precision to save memory
print("Loading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=False  # Allow downloading missing files
)
print("✓ Model loaded")

# Simple test
prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\nPrompt: {prompt}")
print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        do_sample=True
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Output: {result}")