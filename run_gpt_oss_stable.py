#!/usr/bin/env python3
"""
Stable GPT-OSS-20B loader for RTX 3090 using 4-bit quantization
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Skip MXFP4 since RTX 3090 (Ampere) doesn't support it anyway
os.environ['TRANSFORMERS_NO_MXFP4'] = '1'

print("=" * 60)
print("GPT-OSS-20B Stable Loader (4-bit Quantization)")
print("=" * 60)

# Check environment
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {memory_gb:.2f}GB")

# Check critical imports
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("✗ Triton not available")

try:
    import bitsandbytes as bnb
    print("✓ BitsAndBytes available")
except ImportError:
    print("✗ BitsAndBytes not available - install with: pip install bitsandbytes")
    exit(1)

print("\n" + "=" * 60)
print("Loading Model...")
print("=" * 60)

model_id = "openai/gpt-oss-20b"

try:
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Configure 4-bit quantization (NF4 - best for inference)
    print("\n2. Configuring 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use NF4 (normal float 4) - best for inference
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
        bnb_4bit_use_double_quant=True,  # Further compress the quantization constants
    )
    print("✓ 4-bit NF4 quantization configured")
    print("  - Expected memory usage: ~10-12GB")
    
    # Load model
    print("\n3. Loading model with 4-bit quantization...")
    print("   Downloading/loading model files...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",  # Automatically place layers
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("✓ Model loaded successfully!")
    
    # Print model info
    print("\n" + "=" * 60)
    print("Model Information:")
    print("=" * 60)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Memory stats
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated_gb:.2f} GB")
        print(f"  Reserved: {reserved_gb:.2f} GB")
        print(f"  Free: {memory_gb - reserved_gb:.2f} GB")
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing Text Generation:")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In quantum computing, the main challenge is",
        "Python is a programming language that",
    ]
    
    for prompt in test_prompts[:1]:  # Start with just one prompt
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        # Generate
        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS: Model is working!")
    print("=" * 60)
    print("\nQuick Usage Example:")
    print("-" * 40)
    print("""
def generate_text(prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use it:
result = generate_text("Explain quantum entanglement in simple terms:")
print(result)
    """)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Troubleshooting:")
    print("=" * 60)
    print("1. Check CUDA is working: nvidia-smi")
    print("2. Ensure bitsandbytes is installed: pip install bitsandbytes")
    print("3. Check available disk space for model download (~40GB)")
    print("4. Try clearing cache: rm -rf ~/.cache/huggingface/hub/models--openai--gpt-oss-20b")