#!/usr/bin/env python3
"""
Test few-shot priming to activate Harmony channels
"""

import requests
import json
import time

def test_few_shot_priming():
    """Test if few-shot examples activate channels even in Q4"""
    
    print("="*60)
    print("FEW-SHOT PRIMING TEST FOR HARMONY CHANNELS")
    print("="*60)
    
    url = "http://localhost:11434/api/generate"
    
    # Few-shot prompt with example
    few_shot_prompt = """<|system|>
Reasoning: high
<|end|><|user|>
What is 3 + 4?
<|end|><|assistant|>
<analysis>
I need to add 3 and 4 together.
3 + 4 = 7
</analysis>
<commentary>
This is a simple arithmetic problem that doesn't require tools.
</commentary>
<final>
The answer is 7.
</final>
<|end|><|user|>
What are the first 6 prime numbers?
<|end|>"""
    
    print("\n### Test 1: Few-Shot Priming")
    print("Sending prompt with worked example...")
    
    payload = {
        "model": "gpt-oss-harmony",
        "prompt": few_shot_prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
            "stop": ["<|end|>", "<|user|>"]
        }
    }
    
    start = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        output = result.get('response', '')
        tokens = result.get('eval_count', 0)
        
        # Check for channels
        has_analysis = '<analysis>' in output
        has_commentary = '<commentary>' in output
        has_final = '<final>' in output
        
        print(f"\nResults:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens: {tokens}")
        print(f"  Channels detected:")
        print(f"    <analysis>: {'✅' if has_analysis else '❌'}")
        print(f"    <commentary>: {'✅' if has_commentary else '❌'}")
        print(f"    <final>: {'✅' if has_final else '❌'}")
        
        if has_analysis or has_commentary or has_final:
            print(f"\n✅ SUCCESS: Few-shot priming activated channels!")
        else:
            print(f"\n❌ Channels not activated")
        
        print(f"\nResponse preview:")
        print("-"*40)
        print(output[:500])
    
    # Test 2: Tool-calling with few-shot
    print("\n" + "="*60)
    print("### Test 2: Few-Shot Tool Calling")
    
    tool_few_shot = """<|system|>
Reasoning: high
Tools: [{"name":"python","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}]
<|end|><|user|>
Calculate 5 factorial.
<|end|><|assistant|>
<analysis>
I need to calculate 5! which is 5 × 4 × 3 × 2 × 1.
</analysis>
<commentary>
CALL_TOOL: {"name": "python", "arguments": {"code": "import math; print(math.factorial(5))"}}
</commentary>
<final>
5! = 120
</final>
<|end|><|user|>
Calculate the sum of the first 10 Fibonacci numbers.
<|end|>"""
    
    payload = {
        "model": "gpt-oss-harmony",
        "prompt": tool_few_shot,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
            "stop": ["<|end|>", "<|user|>"]
        }
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        output = result.get('response', '')
        
        has_call_tool = 'CALL_TOOL' in output
        has_json = '"name"' in output and '"python"' in output
        has_commentary = '<commentary>' in output
        
        print(f"\nResults:")
        print(f"  CALL_TOOL found: {'✅' if has_call_tool else '❌'}")
        print(f"  JSON structure: {'✅' if has_json else '❌'}")
        print(f"  <commentary> tag: {'✅' if has_commentary else '❌'}")
        
        if has_call_tool:
            print(f"\n✅ SUCCESS: Few-shot priming activated tool calling!")
            # Extract the tool call
            if 'CALL_TOOL' in output:
                lines = output.split('\n')
                for line in lines:
                    if 'CALL_TOOL' in line:
                        print(f"\nExtracted tool call:")
                        print(f"  {line}")
        
        print(f"\nResponse preview:")
        print("-"*40)
        print(output[:500])
    
    # Test 3: Variable reasoning with few-shot
    print("\n" + "="*60)
    print("### Test 3: Variable Reasoning with Few-Shot")
    
    for level in ['low', 'medium', 'high']:
        prompt = f"""<|system|>
Reasoning: {level}
<|end|><|user|>
What is 10 × 10?
<|end|><|assistant|>
<analysis>
10 × 10 = 100
</analysis>
<final>
100
</final>
<|end|><|user|>
What is 15% of 200?
<|end|>"""
        
        payload = {
            "model": "gpt-oss-harmony",
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
                "stop": ["<|end|>", "<|user|>"]
            }
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            output = result.get('response', '')
            tokens = result.get('eval_count', 0)
            
            print(f"\n{level.upper()} reasoning: {tokens} tokens")
            has_analysis = '<analysis>' in output
            if has_analysis:
                print(f"  ✅ Uses <analysis> channel")
            print(f"  Preview: {output[:100]}...")

if __name__ == "__main__":
    test_few_shot_priming()