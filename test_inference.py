#!/usr/bin/env python3
"""Test GPT-OSS-20B model inference"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    print("=" * 60)
    print("Testing GPT-OSS-20B Model Inference")
    print("=" * 60)
    
    # Model path - try local files first
    model_path = "/home/tdeshane/gpt-oss/models"
    
    print(f"\n1. Loading tokenizer...")
    try:
        # Try loading from HuggingFace hub for tokenizer
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        print("   ✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading tokenizer: {e}")
        return
    
    print(f"\n2. Loading model from {model_path}...")
    print("   This may take a while for a 20B parameter model...")
    
    try:
        start_time = time.time()
        
        # Try loading model from local files
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use fp16 to save memory
            device_map="auto",  # Automatically distribute across available devices
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"   ✓ Model loaded successfully in {load_time:.2f} seconds")
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {num_params:,}")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        print("\n   Trying to load from HuggingFace hub instead...")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("   ✓ Model loaded from HuggingFace hub")
        except Exception as e2:
            print(f"   ✗ Failed to load model: {e2}")
            return
    
    print("\n3. Running inference test...")
    
    # Test prompt
    prompt = "The future of artificial intelligence is"
    print(f"   Prompt: '{prompt}'")
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        print("   Generating text...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        gen_time = time.time() - start_time
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n   Generated text ({gen_time:.2f}s):")
        print("   " + "=" * 50)
        print(f"   {generated_text}")
        print("   " + "=" * 50)
        
        # Calculate tokens per second
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = num_tokens / gen_time
        print(f"\n   Performance: {tokens_per_sec:.2f} tokens/second")
        
    except Exception as e:
        print(f"   ✗ Error during inference: {e}")
        return
    
    print("\n✓ Inference test completed successfully!")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\n   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()