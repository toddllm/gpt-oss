#!/usr/bin/env python3
"""
Run GPT-20B model with MXFP4 quantization on RTX 3090
Optimized for 24GB VRAM using 4-bit quantization
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import triton
import triton.language as tl
from typing import Optional
import gc
import os

print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

@triton.jit
def mxfp4_quantize_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """MXFP4 quantization kernel for model weights"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute block-wise scale
    abs_max = tl.max(tl.abs(x))
    scale = abs_max / 7.0 + 1e-8
    
    # Quantize to 4-bit
    x_scaled = x / scale
    quantized = tl.where(x_scaled >= 0,
                         tl.floor(x_scaled + 0.5),
                         tl.ceil(x_scaled - 0.5))
    quantized = tl.minimum(7.0, tl.maximum(-7.0, quantized))
    
    # Store quantized values and scale
    tl.store(output_ptr + offsets, quantized, mask=mask)
    if pid == 0:
        tl.store(scale_ptr, scale)

class MXFP4Linear(nn.Module):
    """4-bit quantized linear layer using Triton"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights in 4-bit format (using int8 for simplicity)
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def quantize_weights(self, weight):
        """Quantize FP32 weights to MXFP4"""
        with torch.cuda.device(weight.device):
            for i in range(self.out_features):
                row = weight[i].contiguous()
                quantized = torch.zeros_like(row, dtype=torch.int8)
                scale = torch.zeros(1, device=weight.device)
                
                BLOCK_SIZE = 256
                n_elements = self.in_features
                grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
                
                mxfp4_quantize_kernel[grid,](
                    row, quantized, scale,
                    n_elements, BLOCK_SIZE=BLOCK_SIZE
                )
                
                self.weight_quantized[i] = quantized
                self.weight_scale[i] = scale.item()
    
    def forward(self, x):
        # Dequantize weights on-the-fly
        weight = self.weight_quantized.float() * self.weight_scale.unsqueeze(1)
        output = torch.nn.functional.linear(x, weight, self.bias)
        return output

def quantize_model(model):
    """Replace linear layers with MXFP4 quantized versions"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create quantized layer
            quantized = MXFP4Linear(module.in_features, module.out_features, module.bias is not None)
            quantized.quantize_weights(module.weight.data)
            if module.bias is not None:
                quantized.bias.data = module.bias.data
            
            # Replace the module
            setattr(model, name, quantized)
        else:
            quantize_model(module)

def load_model_quantized(model_name: str = "EleutherAI/gpt-neox-20b"):
    """
    Load a 20B parameter model with 4-bit quantization
    
    Memory estimation:
    - FP32: 20B * 4 bytes = 80GB (won't fit)
    - FP16: 20B * 2 bytes = 40GB (won't fit) 
    - INT8: 20B * 1 byte = 20GB (tight fit)
    - FP4: 20B * 0.5 bytes = 10GB (comfortable fit with room for activations)
    """
    
    print(f"\n{'='*60}")
    print(f"Loading {model_name} with MXFP4 quantization...")
    print(f"{'='*60}\n")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Try using bitsandbytes 4-bit quantization if available
        try:
            import bitsandbytes as bnb
            print("Using bitsandbytes 4-bit quantization...")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
        except ImportError:
            print("bitsandbytes not available, using custom Triton quantization...")
            
            # Load model in FP16 first to save memory
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu",  # Load to CPU first
                trust_remote_code=True
            )
            
            # Quantize the model
            print("Quantizing model weights to 4-bit...")
            quantize_model(model)
            
            # Move to GPU
            model = model.cuda()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Print memory usage
        print(f"\nMemory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Fallback to smaller model
        print("\nTrying smaller alternative: EleutherAI/gpt-j-6b")
        return load_model_quantized("EleutherAI/gpt-j-6b")

def generate_text(model, tokenizer, prompt: str, max_length: int = 100):
    """Generate text using the quantized model"""
    
    print(f"\n{'='*60}")
    print("Generating text...")
    print(f"{'='*60}\n")
    print(f"Prompt: {prompt}\n")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated:\n{generated}")
    
    return generated

def main():
    """Main function to run the quantized GPT model"""
    
    # Set environment for optimal performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Model options (in order of preference for 24GB VRAM)
    model_options = [
        "EleutherAI/gpt-neox-20b",      # 20B parameters
        "EleutherAI/gpt-j-6b",           # 6B parameters (fallback)
        "facebook/opt-13b",              # 13B parameters
        "bigscience/bloom-7b1"           # 7B parameters
    ]
    
    # Try to load the largest model that fits
    model = None
    tokenizer = None
    
    for model_name in model_options:
        try:
            print(f"\nAttempting to load: {model_name}")
            model, tokenizer = load_model_quantized(model_name)
            if model is not None:
                print(f"✓ Successfully loaded {model_name}")
                break
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    if model is None:
        print("\n✗ Could not load any model. Please check your setup.")
        return
    
    # Test generation
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The key to happiness is"
    ]
    
    for prompt in test_prompts:
        generate_text(model, tokenizer, prompt, max_length=50)
        print("\n" + "-"*60)
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*60)
    
    while True:
        user_prompt = input("\nEnter prompt: ")
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            generate_text(model, tokenizer, user_prompt, max_length=100)
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPT-20B with MXFP4 Quantization on RTX 3090")
    print("="*60)
    
    # First check if we need to install dependencies
    try:
        import transformers
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers accelerate")
    
    try:
        import bitsandbytes
        print("✓ bitsandbytes available for 4-bit quantization")
    except ImportError:
        print("⚠ bitsandbytes not found. Install with: pip install bitsandbytes")
        print("  Will use custom Triton quantization as fallback")
    
    main()