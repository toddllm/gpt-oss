#!/usr/bin/env python3
"""
Test script for gpt-oss-20b API (works with vLLM or Ollama)
"""

import requests
import json
import sys
import time

def test_vllm_api():
    """Test vLLM OpenAI-compatible API"""
    
    url = "http://localhost:8000/v1/chat/completions"
    
    print("Testing vLLM API at", url)
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain VRAM like I'm five."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n✓ vLLM API Response:")
        print("-" * 40)
        print(result['choices'][0]['message']['content'])
        print("-" * 40)
        
        # Print usage stats
        if 'usage' in result:
            print(f"\nTokens used:")
            print(f"  Prompt: {result['usage']['prompt_tokens']}")
            print(f"  Completion: {result['usage']['completion_tokens']}")
            print(f"  Total: {result['usage']['total_tokens']}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to vLLM server at", url)
        print("   Make sure to run: ./run_vllm.sh")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_api():
    """Test Ollama API"""
    
    url = "http://localhost:11434/api/chat"
    
    print("\nTesting Ollama API at", url)
    
    payload = {
        "model": "gpt-oss:20b",
        "messages": [
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n✓ Ollama API Response:")
        print("-" * 40)
        print(result['message']['content'])
        print("-" * 40)
        
        # Print timing info
        if 'total_duration' in result:
            total_ms = result['total_duration'] / 1_000_000
            eval_ms = result.get('eval_duration', 0) / 1_000_000
            tokens = result.get('eval_count', 0)
            
            print(f"\nPerformance:")
            print(f"  Total time: {total_ms:.0f}ms")
            print(f"  Generation time: {eval_ms:.0f}ms")
            if tokens > 0 and eval_ms > 0:
                print(f"  Tokens/sec: {tokens * 1000 / eval_ms:.1f}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama server at", url)
        print("   Make sure to run: ./run_ollama.sh")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def benchmark_api(api_type="vllm"):
    """Run a performance benchmark"""
    
    if api_type == "vllm":
        url = "http://localhost:8000/v1/chat/completions"
        model_name = "openai/gpt-oss-20b"
    else:
        url = "http://localhost:11434/api/chat"
        model_name = "gpt-oss:20b"
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {api_type.upper()} API")
    print(f"{'='*60}")
    
    prompts = [
        "Write a Python function to calculate fibonacci numbers.",
        "Explain the difference between CPU and GPU.",
        "What are the benefits of open source software?",
    ]
    
    total_tokens = 0
    total_time = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}: {prompt[:50]}...")
        
        if api_type == "vllm":
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 100
            }
        else:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        
        start = time.time()
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start
            
            if api_type == "vllm":
                tokens = result['usage']['completion_tokens']
            else:
                tokens = result.get('eval_count', 50)  # estimate
            
            tokens_per_sec = tokens / elapsed
            
            print(f"  → {tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
            
            total_tokens += tokens
            total_time += elapsed
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    if total_time > 0:
        print(f"\n{'='*60}")
        print(f"Average: {total_tokens/total_time:.1f} tokens/second")
        print(f"{'='*60}")

def main():
    print("="*60)
    print("GPT-OSS-20B API Test Suite")
    print("="*60)
    
    # Test both APIs
    vllm_ok = test_vllm_api()
    ollama_ok = test_ollama_api()
    
    if not vllm_ok and not ollama_ok:
        print("\n⚠️  No API servers detected.")
        print("Please start one of the following:")
        print("  • vLLM: ./run_vllm.sh")
        print("  • Ollama: ./run_ollama.sh")
        sys.exit(1)
    
    # Run benchmark on available API
    if vllm_ok:
        benchmark_api("vllm")
    elif ollama_ok:
        benchmark_api("ollama")

if __name__ == "__main__":
    main()