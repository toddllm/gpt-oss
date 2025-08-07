#!/usr/bin/env python3
"""
Test GPT-OSS with raw prompts bypassing Ollama's template
"""

import subprocess
import json

def test_raw_harmony():
    """Test with native Harmony format"""
    
    print("="*60)
    print("TESTING RAW HARMONY FORMAT")
    print("="*60)
    
    # Test 1: Native Harmony format
    harmony_prompt = """<|system|>
Reasoning: high
<|end|><|user|>
Solve: What is 15% of 240? Use the following format:
<analysis>Show your step-by-step reasoning here</analysis>
<commentary>Any meta-commentary or tool calls</commentary>
<final>Your final answer to the user</final>
<|end|>"""
    
    print("\n### Test 1: Native Harmony Format")
    print("Prompt:", harmony_prompt[:100] + "...")
    
    result = subprocess.run(
        ['ollama', 'run', 'gpt-oss:latest', '--raw'],
        input=harmony_prompt,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    response = result.stdout
    print("\nResponse:", response[:500])
    
    # Check for channels
    has_analysis = '<analysis>' in response
    has_commentary = '<commentary>' in response
    has_final = '<final>' in response
    
    print(f"\nChannels detected:")
    print(f"  <analysis>: {'✓' if has_analysis else '✗'}")
    print(f"  <commentary>: {'✓' if has_commentary else '✗'}")
    print(f"  <final>: {'✓' if has_final else '✗'}")
    
    # Test 2: Variable reasoning
    print("\n" + "="*60)
    print("### Test 2: Variable Reasoning Levels")
    
    for level in ['low', 'medium', 'high']:
        prompt = f"""<|system|>
Reasoning: {level}
<|end|><|user|>
Calculate the factorial of 5. Show your work.
<|end|>"""
        
        print(f"\n--- {level.upper()} reasoning ---")
        
        result = subprocess.run(
            ['ollama', 'run', 'gpt-oss:latest', '--raw'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        response = result.stdout
        response_len = len(response)
        print(f"Response length: {response_len} chars")
        print(f"Preview: {response[:200]}...")
    
    # Test 3: Tool calling with proper schema
    print("\n" + "="*60)
    print("### Test 3: Tool Calling with Schema")
    
    tool_prompt = """<|system|>
Tools: [{"name":"python","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}]
Reasoning: medium
<|end|><|user|>
Calculate the sum of the first 10 prime numbers using Python.
<|end|>"""
    
    result = subprocess.run(
        ['ollama', 'run', 'gpt-oss:latest', '--raw'],
        input=tool_prompt,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    response = result.stdout
    print("Response:", response[:500])
    
    # Check for tool patterns
    has_call_tool = 'CALL_TOOL' in response
    has_json = '{' in response and '"name"' in response
    has_python = 'python' in response.lower()
    
    print(f"\nTool indicators:")
    print(f"  CALL_TOOL: {'✓' if has_call_tool else '✗'}")
    print(f"  JSON structure: {'✓' if has_json else '✗'}")
    print(f"  Python mentioned: {'✓' if has_python else '✗'}")

if __name__ == "__main__":
    test_raw_harmony()