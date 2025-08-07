#!/usr/bin/env python3
"""
Working GPT-OSS-20B loader with proper quantization handling
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Skip MXFP4 since RTX 3090 doesn't support it
os.environ['TRANSFORMERS_NO_MXFP4'] = '1'

print("=" * 60)
print("GPT-OSS-20B Working Loader")
print("=" * 60)

# Check environment
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {memory_gb:.2f}GB")

# Check imports
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("✗ Triton not available")

try:
    import bitsandbytes as bnb
    print("✓ BitsAndBytes available")
    HAS_BNB = True
except ImportError:
    print("✗ BitsAndBytes not available")
    HAS_BNB = False

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
    
    # Try different loading strategies
    print("\n2. Attempting to load model...")
    
    # Strategy 1: Try with 4-bit quantization if bitsandbytes is available
    if HAS_BNB:
        try:
            print("   Trying 4-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            print("✓ Model loaded with 4-bit quantization!")
            quantization_type = "4-bit"
        except Exception as e:
            print(f"   4-bit failed: {str(e)[:100]}")
            model = None
    
    # Strategy 2: Try with 8-bit quantization
    if model is None and HAS_BNB:
        try:
            print("   Trying 8-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            print("✓ Model loaded with 8-bit quantization!")
            quantization_type = "8-bit"
        except Exception as e:
            print(f"   8-bit failed: {str(e)[:100]}")
            model = None
    
    # Strategy 3: Try with bfloat16
    if model is None:
        try:
            print("   Trying bfloat16 (no quantization)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            print("✓ Model loaded with bfloat16!")
            quantization_type = "bfloat16"
        except Exception as e:
            print(f"   bfloat16 failed: {str(e)[:100]}")
            model = None
    
    # Strategy 4: Try with float16
    if model is None:
        try:
            print("   Trying float16...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            print("✓ Model loaded with float16!")
            quantization_type = "float16"
        except Exception as e:
            print(f"   float16 failed: {str(e)[:100]}")
            raise RuntimeError("Failed to load model with any strategy")
    
    # Print model info
    print("\n" + "=" * 60)
    print("Model Information:")
    print("=" * 60)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Quantization: {quantization_type}")
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
    
    test_prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: '{test_prompt}'")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
    print(f"✓ SUCCESS: Model loaded with {quantization_type} and working!")
    print("=" * 60)
    
    # Save configuration for future use
    config_summary = f"""
# Working Configuration for GPT-OSS-20B
Model: {model_id}
Quantization: {quantization_type}
PyTorch: {torch.__version__}
Triton: {triton.__version__ if 'triton' in locals() else 'N/A'}
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
Memory Used: {allocated_gb:.2f} GB

To use in your code:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_id}",
    {"load_in_4bit=True," if quantization_type == "4-bit" else "load_in_8bit=True," if quantization_type == "8-bit" else ""}
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.{str(model.dtype).replace('torch.', '')},
)
```
"""
    
    with open("working_config.txt", "w") as f:
        f.write(config_summary)
    print(f"\nConfiguration saved to working_config.txt")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Troubleshooting:")
    print("=" * 60)
    print("1. Check GPU memory: nvidia-smi")
    print("2. Clear cache: rm -rf ~/.cache/huggingface/")
    print("3. Install bitsandbytes: pip install bitsandbytes")
    print("4. Check disk space: df -h")