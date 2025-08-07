#!/usr/bin/env python3
"""
Test transformers with MXFP4 quantization for gpt-oss-20b
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add triton_kernels to path
sys.path.insert(0, '.')

print("=" * 60)
print("Testing Transformers with MXFP4 Quantization")
print("=" * 60)

# Check versions
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {memory_gb:.2f}GB")

# Check triton
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("✗ Triton not available")

# Check triton_kernels
try:
    import triton_kernels
    print("✓ triton_kernels available")
except ImportError:
    print("✗ triton_kernels not available")

# Check bitsandbytes
try:
    import bitsandbytes as bnb
    print("✓ bitsandbytes available")
except ImportError:
    print("✗ bitsandbytes not available")

print("\n" + "=" * 60)
print("Loading gpt-oss-20b with MXFP4 quantization...")
print("=" * 60)

model_id = "openai/gpt-oss-20b"

try:
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("✓ Tokenizer loaded")
    
    # Configure MXFP4 quantization
    print("\n2. Configuring MXFP4 quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="mxfp4",  # Use MXFP4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("✓ MXFP4 configuration created")
    
    # Load model with quantization
    print("\n3. Loading model with MXFP4 quantization...")
    print("   This may take a few minutes...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print("✓ Model loaded successfully!")
    
    # Print model info
    print("\n" + "=" * 60)
    print("Model Information:")
    print("=" * 60)
    print(f"Model type: {type(model)}")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    # Calculate memory usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated_gb:.2f} GB")
        print(f"  Reserved: {reserved_gb:.2f} GB")
        print(f"  Available: {memory_gb - reserved_gb:.2f} GB")
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing Text Generation:")
    print("=" * 60)
    
    test_prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {test_prompt}")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS: Model loaded and inference working!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Troubleshooting Tips:")
    print("=" * 60)
    print("1. Ensure triton >= 3.4.0 is installed")
    print("2. Ensure triton_kernels is available")
    print("3. Check GPU memory availability")
    print("4. Try with smaller batch size or sequence length")