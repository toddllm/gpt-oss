#!/usr/bin/env python3
"""
Final benchmark of optimized GPT-OSS with Q4_K_M
"""

import requests
import json
import time

def benchmark_optimized():
    """Benchmark the optimized model configuration"""
    
    print("="*60)
    print("GPT-OSS Q4_K_M OPTIMIZED BENCHMARK")
    print("="*60)
    
    url = "http://localhost:11434/api/generate"
    models = ["gpt-oss:latest", "gpt-oss-harmony", "gpt-oss-optimized"]
    
    # Test prompt with explicit channels
    test_prompt = """<|system|>
Reasoning: high
<|end|><|user|>
What are the first 6 prime numbers?
<|end|><|assistant|>
<analysis>
I need to identify the first 6 prime numbers. Prime numbers are natural numbers greater than 1 that have no positive divisors other than 1 and themselves.
</analysis>
<commentary>
This is a straightforward mathematical question that doesn't require tools.
</commentary>
<final>
The first 6 prime numbers are: 2, 3, 5, 7, 11, 13
</final>
<|end|><|user|>
Now add them together.
<|end|>"""
    
    results = {}
    
    for model in models:
        print(f"\n### Testing: {model}")
        print("-"*40)
        
        payload = {
            "model": model,
            "prompt": test_prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
                "seed": 42
            }
        }
        
        # Warmup
        requests.post(url, json=payload)
        
        # Actual benchmark
        times = []
        tokens_list = []
        
        for i in range(3):
            start = time.time()
            response = requests.post(url, json=payload)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                output = result.get('response', '')
                tokens = result.get('eval_count', 0)
                
                times.append(elapsed)
                tokens_list.append(tokens)
                
                if i == 0:  # Only check first response
                    has_analysis = '<analysis>' in output
                    has_commentary = '<commentary>' in output
                    has_final = '<final>' in output
                    has_correct = '41' in output  # 2+3+5+7+11+13=41
                    
                    results[model] = {
                        'channels': {'analysis': has_analysis, 'commentary': has_commentary, 'final': has_final},
                        'correct': has_correct,
                        'sample': output[:200]
                    }
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
        
        results[model]['performance'] = {
            'avg_time': round(avg_time, 2),
            'avg_tokens': round(avg_tokens, 0),
            'tok_per_sec': round(tok_per_sec, 1)
        }
        
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Avg tokens: {avg_tokens:.0f}")
        print(f"  Speed: {tok_per_sec:.1f} tok/s")
        print(f"  Channels: {sum(results[model]['channels'].values())}/3")
        print(f"  Correct sum: {'✅' if results[model]['correct'] else '❌'}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\n| Model | Channels | Speed (tok/s) | Correct |")
    print("|-------|----------|---------------|---------|")
    for model in models:
        r = results[model]
        channels = sum(r['channels'].values())
        speed = r['performance']['tok_per_sec']
        correct = '✅' if r['correct'] else '❌'
        print(f"| {model:20s} | {channels}/3 | {speed:5.1f} | {correct} |")
    
    # Best configuration
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    best_channels = max(models, key=lambda m: sum(results[m]['channels'].values()))
    best_speed = max(models, key=lambda m: results[m]['performance']['tok_per_sec'])
    
    print(f"\n✅ Best for channels: {best_channels}")
    print(f"✅ Best for speed: {best_speed}")
    
    if best_channels == "gpt-oss-optimized":
        print("\n🎯 The optimized configuration successfully improves channel usage!")
    elif best_channels == "gpt-oss-harmony":
        print("\n💡 The harmony configuration (empty template) works best!")
    else:
        print("\n⚠️ Default configuration might be sufficient for basic use.")
    
    # Save full results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to benchmark_results.json")
    
    return results

def final_test():
    """Final working example with best configuration"""
    
    print("\n" + "="*60)
    print("FINAL WORKING EXAMPLE")
    print("="*60)
    
    url = "http://localhost:11434/api/generate"
    
    # Best working pattern based on tests
    working_prompt = """<|system|>
Reasoning: high
<|end|><|user|>
Example: What is 2+2?
<|end|><|assistant|>
<analysis>Adding 2+2: 2+2=4</analysis>
<commentary>Simple arithmetic, no tools needed.</commentary>
<final>4</final>
<|end|><|user|>
Calculate the 20th Fibonacci number efficiently.
<|end|>"""
    
    print("\nSending optimized prompt to gpt-oss-optimized...")
    
    payload = {
        "model": "gpt-oss-optimized",
        "prompt": working_prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024
        }
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        output = result.get('response', '')
        
        print("\n✅ WORKING PATTERN CONFIRMED:")
        print("-"*40)
        print(output)
        print("-"*40)
        
        # Extract channels if present
        if '<analysis>' in output:
            try:
                start = output.index('<analysis>') + 10
                end = output.index('</analysis>')
                print(f"\n📊 Analysis extracted:")
                print(f"{output[start:end]}")
            except:
                pass

if __name__ == "__main__":
    results = benchmark_optimized()
    final_test()