#!/usr/bin/env python3
"""
Test GPT-OSS with Ollama API using raw prompts
"""

import requests
import json

def test_ollama_api():
    """Test with Ollama API for more control"""
    
    print("="*60)
    print("TESTING OLLAMA API WITH RAW PROMPTS")
    print("="*60)
    
    url = "http://localhost:11434/api/generate"
    
    # Test 1: Try without template
    print("\n### Test 1: Raw prompt without template")
    
    payload = {
        "model": "gpt-oss:latest",
        "prompt": "<|system|>\nReasoning: high\n<|end|><|user|>\nWhat is 15% of 240?\n<|end|>",
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 512
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        output = result.get('response', '')
        print(f"Response: {output[:300]}...")
        print(f"Total tokens: {result.get('eval_count', 0)}")
    else:
        print(f"Error: {response.status_code}")
    
    # Test 2: Try with explicit channels
    print("\n### Test 2: Explicit channel format")
    
    payload = {
        "model": "gpt-oss:latest",
        "prompt": """Please respond using this format:
<analysis>Your reasoning here</analysis>
<commentary>Any comments</commentary>
<final>Your answer</final>

Question: What is 15% of 240?""",
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        output = result.get('response', '')
        
        has_analysis = '<analysis>' in output
        has_commentary = '<commentary>' in output
        has_final = '<final>' in output
        
        print(f"Channels found:")
        print(f"  <analysis>: {'✓' if has_analysis else '✗'}")
        print(f"  <commentary>: {'✓' if has_commentary else '✗'}")
        print(f"  <final>: {'✓' if has_final else '✗'}")
        print(f"\nResponse preview: {output[:300]}...")
    
    # Test 3: System message with reasoning level
    print("\n### Test 3: System message with reasoning levels")
    
    for level in ['low', 'medium', 'high']:
        payload = {
            "model": "gpt-oss:latest",
            "prompt": f"Calculate factorial of 5",
            "system": f"Reasoning: {level}",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            output = result.get('response', '')
            tokens = result.get('eval_count', 0)
            print(f"\n{level.upper()}: {tokens} tokens")
            print(f"  Preview: {output[:150]}...")

if __name__ == "__main__":
    test_ollama_api()