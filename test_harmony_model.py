#!/usr/bin/env python3
"""
Test GPT-OSS Harmony model with native format
"""

import subprocess
import requests
import json

def test_harmony_model():
    """Test the new Harmony model without templates"""
    
    print("="*60)
    print("TESTING GPT-OSS-HARMONY MODEL")
    print("="*60)
    
    # Test 1: Native Harmony format with subprocess
    print("\n### Test 1: Native Harmony Format (subprocess)")
    
    harmony_prompt = """<|system|>
Reasoning: high
<|end|><|user|>
What are the first 6 prime numbers? Use these channels:
<analysis>Show your reasoning</analysis>
<commentary>Any comments</commentary>
<final>Your answer</final>
<|end|>"""
    
    result = subprocess.run(
        ['ollama', 'run', 'gpt-oss-harmony'],
        input=harmony_prompt,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    response = result.stdout
    print(f"Response length: {len(response)} chars")
    print(f"Response: {response[:500]}...")
    
    # Check for channels
    has_analysis = '<analysis>' in response
    has_commentary = '<commentary>' in response
    has_final = '<final>' in response
    has_system_token = '<|system|>' in response or '<|end|>' in response
    
    print(f"\nChannels detected:")
    print(f"  <analysis>: {'✓' if has_analysis else '✗'}")
    print(f"  <commentary>: {'✓' if has_commentary else '✗'}")
    print(f"  <final>: {'✓' if has_final else '✗'}")
    print(f"  System tokens: {'✓' if has_system_token else '✗'}")
    
    # Test 2: Using API with raw prompt
    print("\n### Test 2: API with Raw Prompt")
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "gpt-oss-harmony",
        "prompt": "<|system|>\nReasoning: high\n<|end|><|user|>\nWhat is 17 * 19? Show your work.\n<|end|>",
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
        tokens = result.get('eval_count', 0)
        
        print(f"Response tokens: {tokens}")
        print(f"Response: {output[:500]}...")
        
        # Check for reasoning patterns
        has_calculation = '*' in output or '17' in output or '19' in output
        has_answer = '323' in output
        
        print(f"\nReasoning indicators:")
        print(f"  Shows calculation: {'✓' if has_calculation else '✗'}")
        print(f"  Correct answer (323): {'✓' if has_answer else '✗'}")
    
    # Test 3: Variable reasoning levels
    print("\n### Test 3: Variable Reasoning Levels")
    
    for level in ['low', 'medium', 'high']:
        payload = {
            "model": "gpt-oss-harmony",
            "prompt": f"<|system|>\nReasoning: {level}\n<|end|><|user|>\nWhat is the capital of France?\n<|end|>",
            "stream": False,
            "raw": True,
            "options": {
                "temperature": 0.3,
                "num_predict": 256
            }
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            output = result.get('response', '')
            tokens = result.get('eval_count', 0)
            
            print(f"\n{level.upper()}: {tokens} tokens")
            print(f"  Response: {output[:150]}...")
    
    # Test 4: Tool calling with native format
    print("\n### Test 4: Tool Calling with Native Format")
    
    tool_prompt = """<|system|>
Reasoning: high
Tools: [{"name": "python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}]
<|end|><|user|>
Calculate the 10th Fibonacci number. Use the python tool in a <commentary> block:
<commentary>
CALL_TOOL: {"name": "python", "arguments": {"code": "your_code"}}
</commentary>
<|end|>"""
    
    payload = {
        "model": "gpt-oss-harmony",
        "prompt": tool_prompt,
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
        
        has_commentary = '<commentary>' in output
        has_call_tool = 'CALL_TOOL' in output
        has_fibonacci = 'fibonacci' in output.lower() or 'fib' in output.lower()
        
        print(f"Tool calling indicators:")
        print(f"  <commentary> tag: {'✓' if has_commentary else '✗'}")
        print(f"  CALL_TOOL: {'✓' if has_call_tool else '✗'}")
        print(f"  Fibonacci mentioned: {'✓' if has_fibonacci else '✗'}")
        print(f"\nResponse: {output[:500]}...")

if __name__ == "__main__":
    test_harmony_model()