#!/usr/bin/env python3
"""
Automatic test that runs when download completes
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("\n" + "="*60)
print("DOWNLOAD COMPLETE - RUNNING AUTOMATIC TEST")
print("="*60)

model_id = "openai/gpt-oss-20b"

try:
    # Environment check
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # Try Flash Attention
    try:
        import flash_attn
        attn = "flash_attention_2"
        print("✓ Flash Attention available")
    except:
        attn = "sdpa"
        print("✗ Flash Attention not found, using SDPA")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model (8-bit quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn,
    )
    
    print("✓ Model loaded successfully!")
    
    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory Used: {allocated:.2f}GB")
    
    # Test generation
    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60)
    
    test_prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: '{test_prompt}'")
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(test_prompt):].strip()
    
    print(f"Response: {generated}")
    
    print("\n" + "="*60)
    print("✓ SUCCESS - MODEL IS WORKING!")
    print("="*60)
    
    # Save success flag
    with open("test_success.flag", "w") as f:
        f.write("SUCCESS")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
