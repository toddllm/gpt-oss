#!/usr/bin/env python3
"""
Run gpt-oss-20b using Transformers library
Flexible approach with multiple quantization options
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import time

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.2f} GB")
        
        if vram_gb < 16:
            print("⚠️  Warning: Less than 16GB VRAM detected. Will use 4-bit quantization.")
            return "4bit"
        elif vram_gb < 24:
            print("ℹ️  16-24GB VRAM detected. Will use 8-bit quantization.")
            return "8bit"
        else:
            print("✓ 24GB+ VRAM detected. Can run in FP16 with MXFP4.")
            return "fp16"
    else:
        print("❌ No CUDA GPU detected!")
        sys.exit(1)

def load_model(quantization="auto"):
    """Load gpt-oss-20b with specified quantization"""
    
    model_id = "openai/gpt-oss-20b"
    
    print(f"\nLoading {model_id}...")
    print(f"Quantization: {quantization}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure model loading based on quantization
    load_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # Try to use Flash Attention 2
    try:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("✓ Using Flash Attention 2")
    except:
        print("ℹ️  Flash Attention 2 not available, using default attention")
    
    if quantization == "4bit":
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_quant_type"] = "nf4"
        load_kwargs["bnb_4bit_use_double_quant"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    elif quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"\nMemory usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return model, tokenizer

def chat_loop(model, tokenizer):
    """Interactive chat loop"""
    
    print("\n" + "="*60)
    print("GPT-OSS-20B Interactive Chat")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("="*60 + "\n")
    
    messages = []
    
    while True:
        # Get user input
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            messages = []
            print("🔄 Conversation cleared!")
            continue
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        # Generate response
        print("\n🤖 GPT-OSS: ", end="", flush=True)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=None  # Could add TextStreamer for streaming output
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        print(response)
        
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response})
        
        # Print generation stats
        gen_time = time.time() - start_time
        tokens_generated = outputs.shape[-1] - inputs.shape[-1]
        tokens_per_sec = tokens_generated / gen_time
        print(f"\n📊 Generated {tokens_generated} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")

def benchmark(model, tokenizer):
    """Run a simple benchmark"""
    
    print("\n" + "="*60)
    print("Running benchmark...")
    print("="*60 + "\n")
    
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What are the key differences between Python and Rust?",
    ]
    
    total_tokens = 0
    total_time = 0
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt[:50]}...")
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize()
        gen_time = time.time() - start
        
        tokens_generated = outputs.shape[-1] - inputs.shape[-1]
        tokens_per_sec = tokens_generated / gen_time
        
        print(f"  → {tokens_generated} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        total_tokens += tokens_generated
        total_time += gen_time
    
    print(f"\nAverage: {total_tokens/total_time:.1f} tokens/second")

def main():
    parser = argparse.ArgumentParser(description="Run gpt-oss-20b with Transformers")
    parser.add_argument("--quantization", choices=["auto", "fp16", "8bit", "4bit"], 
                       default="auto", help="Quantization mode")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run benchmark instead of interactive chat")
    args = parser.parse_args()
    
    # Check requirements and determine quantization
    if args.quantization == "auto":
        args.quantization = check_requirements()
    
    # Load model
    model, tokenizer = load_model(args.quantization)
    
    if args.benchmark:
        benchmark(model, tokenizer)
    else:
        chat_loop(model, tokenizer)

if __name__ == "__main__":
    main()