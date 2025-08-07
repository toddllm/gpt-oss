#!/usr/bin/env python3
"""
Test tool calling capabilities
"""

import ollama
import time
import json
import re

def test_tool_calls():
    """Test tool calling capabilities"""
    
    model = 'gpt-oss:latest'
    
    print("="*60)
    print("TESTING TOOL CALLING CAPABILITIES")
    print("="*60)
    
    # Define tools in the system prompt
    system_prompt = """Reasoning: high
Tools available:
[
  {
    "name": "python",
    "description": "Execute Python code",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string", "description": "Python code to execute"}
      },
      "required": ["code"]
    }
  },
  {
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {"type": "string", "description": "Mathematical expression"}
      },
      "required": ["expression"]
    }
  }
]

When you need to use a tool, format it as:
CALL_TOOL: {"name": "tool_name", "arguments": {...}}

You can also use the <commentary> channel for tool calls."""
    
    test_cases = [
        {
            'query': "Calculate the factorial of 10",
            'expected_tool': 'python',
            'check_for': ['factorial', '3628800', 'def', 'for', 'range']
        },
        {
            'query': "What is 15% of 240?",
            'expected_tool': 'calculator',
            'check_for': ['36', '0.15', '240', '%']
        },
        {
            'query': "Generate the first 10 prime numbers",
            'expected_tool': 'python',
            'check_for': ['prime', '2', '3', '5', '7', '11', '13', '17', '19', '23', '29']
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n### Test {i}: {test['query']}")
        print("-"*60)
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': test['query']}
                ],
                stream=False,
                options={
                    'temperature': 0.7,
                    'num_predict': 2048
                }
            )
            
            elapsed = time.time() - start_time
            content = response['message']['content']
            
            # Check for various tool call patterns
            has_call_tool = 'CALL_TOOL' in content
            has_python_keyword = 'python' in content.lower()
            has_calculator = 'calculator' in content.lower()
            
            # Look for code blocks
            has_code_block = '```' in content
            has_python_code = bool(re.search(r'(def |import |print\(|for |while |if |=)', content))
            
            # Check for JSON-like tool calls
            has_json_pattern = bool(re.search(r'\{.*"name".*:.*"(python|calculator)".*\}', content, re.DOTALL))
            
            # Check for expected content
            found_keywords = [kw for kw in test['check_for'] if kw.lower() in content.lower()]
            
            result = {
                'query': test['query'],
                'expected_tool': test['expected_tool'],
                'has_call_tool': has_call_tool,
                'has_tool_mention': has_python_keyword or has_calculator,
                'has_code': has_python_code,
                'has_code_block': has_code_block,
                'has_json_pattern': has_json_pattern,
                'found_keywords': len(found_keywords),
                'total_keywords': len(test['check_for']),
                'response_length': len(content),
                'time_seconds': round(elapsed, 2)
            }
            
            results.append(result)
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Response length: {len(content)} chars")
            print(f"  Tool call indicators:")
            print(f"    CALL_TOOL found:  {'✓' if has_call_tool else '✗'}")
            print(f"    Tool mentioned:   {'✓' if result['has_tool_mention'] else '✗'}")
            print(f"    Code present:     {'✓' if has_python_code else '✗'}")
            print(f"    Code block:       {'✓' if has_code_block else '✗'}")
            print(f"    JSON pattern:     {'✓' if has_json_pattern else '✗'}")
            print(f"  Keywords found: {len(found_keywords)}/{len(test['check_for'])}")
            
            # Extract and show relevant parts
            if has_call_tool:
                lines = content.split('\n')
                for line in lines:
                    if 'CALL_TOOL' in line:
                        print(f"\n  Tool call found:")
                        print(f"    {line[:200]}...")
                        break
            
            if has_code_block:
                # Extract first code block
                match = re.search(r'```[\w]*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    code = match.group(1)
                    print(f"\n  Code block found:")
                    print(f"    {code[:300]}...")
            elif has_python_code:
                print(f"\n  Python code detected (first 300 chars):")
                print(f"    {content[:300]}...")
                
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'query': test['query'], 'error': str(e)})
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in results if 'error' not in r]
    if successful_tests:
        total = len(successful_tests)
        
        with_call_tool = sum(1 for r in successful_tests if r['has_call_tool'])
        with_tool_mention = sum(1 for r in successful_tests if r['has_tool_mention'])
        with_code = sum(1 for r in successful_tests if r['has_code'])
        with_json = sum(1 for r in successful_tests if r['has_json_pattern'])
        
        print(f"\nTool calling indicators across {total} tests:")
        print(f"  CALL_TOOL pattern:   {with_call_tool}/{total} ({100*with_call_tool/total:.0f}%)")
        print(f"  Tool mentioned:      {with_tool_mention}/{total} ({100*with_tool_mention/total:.0f}%)")
        print(f"  Code present:        {with_code}/{total} ({100*with_code/total:.0f}%)")
        print(f"  JSON pattern:        {with_json}/{total} ({100*with_json/total:.0f}%)")
        
        # Keyword accuracy
        total_found = sum(r['found_keywords'] for r in successful_tests)
        total_expected = sum(r['total_keywords'] for r in successful_tests)
        print(f"\nContent accuracy:")
        print(f"  Keywords found: {total_found}/{total_expected} ({100*total_found/total_expected:.0f}%)")
    
    # Save results
    with open('tool_calls_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to tool_calls_results.json")
    
    return results

if __name__ == "__main__":
    test_tool_calls()