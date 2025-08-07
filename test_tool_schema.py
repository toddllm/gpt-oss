#!/usr/bin/env python3
"""
Test GPT-OSS tool calling with proper schema from training
"""

import requests
import json

def test_tool_schema():
    """Test with exact tool schema from training"""
    
    print("="*60)
    print("TESTING TOOL CALLING WITH TRAINING SCHEMA")
    print("="*60)
    
    url = "http://localhost:11434/api/generate"
    
    # Test 1: Exact training format
    print("\n### Test 1: Training-style tool schema")
    
    tool_schema = {
        "name": "python",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"}
            },
            "required": ["code"]
        }
    }
    
    prompt = f"""Tools: [{json.dumps(tool_schema)}]
Reasoning: medium

When you need to use a tool, format your response as:
<commentary>
CALL_TOOL: {{"name": "python", "arguments": {{"code": "your_code_here"}}}}
</commentary>

User: Calculate the sum of the first 10 prime numbers."""
    
    payload = {
        "model": "gpt-oss:latest",
        "prompt": prompt,
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
        
        # Check for tool call patterns
        has_call_tool = 'CALL_TOOL' in output
        has_commentary = '<commentary>' in output
        has_json_structure = '"name"' in output and '"python"' in output
        has_code = 'prime' in output.lower() or 'def' in output
        
        print(f"Tool call indicators:")
        print(f"  CALL_TOOL found: {'✓' if has_call_tool else '✗'}")
        print(f"  <commentary> tag: {'✓' if has_commentary else '✗'}")
        print(f"  JSON structure: {'✓' if has_json_structure else '✗'}")
        print(f"  Code present: {'✓' if has_code else '✗'}")
        print(f"\nResponse preview: {output[:500]}...")
        
        # Try to extract tool call if present
        if 'CALL_TOOL' in output:
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if 'CALL_TOOL' in line:
                    print(f"\nExtracted tool call:")
                    print(f"  {line}")
                    # Try to parse the JSON
                    try:
                        json_start = line.index('{')
                        json_str = line[json_start:]
                        tool_call = json.loads(json_str)
                        print(f"  Parsed: {tool_call}")
                    except:
                        print(f"  Could not parse JSON")
    
    # Test 2: Multiple tools
    print("\n### Test 2: Multiple tools schema")
    
    tools = [
        {
            "name": "python",
            "description": "Execute Python code",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        },
        {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    ]
    
    prompt = f"""System: You have access to these tools:
{json.dumps(tools, indent=2)}

Use <commentary> tags for tool calls in this format:
<commentary>
CALL_TOOL: {{"name": "tool_name", "arguments": {{...}}}}
</commentary>

User: Search for information about the speed of light, then write Python code to convert it from m/s to km/h."""
    
    payload = {
        "model": "gpt-oss:latest",
        "prompt": prompt,
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
        
        # Count tool mentions
        search_mentioned = 'search' in output.lower()
        python_mentioned = 'python' in output.lower()
        multiple_calls = output.count('CALL_TOOL') > 1
        
        print(f"Multiple tools test:")
        print(f"  Search mentioned: {'✓' if search_mentioned else '✗'}")
        print(f"  Python mentioned: {'✓' if python_mentioned else '✗'}")
        print(f"  Multiple CALL_TOOL: {'✓' if multiple_calls else '✗'}")
        print(f"\nResponse preview: {output[:500]}...")
    
    # Test 3: CoT with tool use
    print("\n### Test 3: Chain-of-thought with tool use")
    
    prompt = """Reasoning: high

You should think step-by-step in <analysis> tags, then use tools in <commentary> tags.

Tools: [{"name": "python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}]

User: What is the 20th Fibonacci number? Calculate it efficiently."""
    
    payload = {
        "model": "gpt-oss:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 2048
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        output = result.get('response', '')
        tokens = result.get('eval_count', 0)
        
        has_analysis = '<analysis>' in output
        has_commentary = '<commentary>' in output
        has_fibonacci = 'fibonacci' in output.lower() or 'fib' in output.lower()
        
        print(f"CoT with tools:")
        print(f"  <analysis> tag: {'✓' if has_analysis else '✗'}")
        print(f"  <commentary> tag: {'✓' if has_commentary else '✗'}")
        print(f"  Fibonacci mentioned: {'✓' if has_fibonacci else '✗'}")
        print(f"  Total tokens: {tokens}")
        print(f"\nResponse preview: {output[:500]}...")

if __name__ == "__main__":
    test_tool_schema()