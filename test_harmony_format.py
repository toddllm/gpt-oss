#!/usr/bin/env python3
"""
Test Harmony chat format with role-based channels
"""

import ollama
import time
import json

def test_harmony_format():
    """Test the Harmony chat format with role-based channels"""
    
    model = 'gpt-oss:latest'
    
    print("="*60)
    print("TESTING HARMONY CHAT FORMAT")
    print("="*60)
    
    # Test with explicit channel instructions
    system_prompt = """Reasoning: high
You should structure your response using these channels:
<analysis> - For your reasoning and thinking process
<commentary> - For any tool calls or meta-commentary  
<final> - For your final answer to the user

Always use all three channels in your response."""
    
    test_queries = [
        "Solve: If 2x + 5 = 13, what is x?",
        "What causes tides on Earth?",
        "Write a Python function to check if a number is prime"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n### Test {i}: {query[:50]}...")
        print("-"*60)
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query}
                ],
                stream=False,
                options={
                    'temperature': 0.7,
                    'num_predict': 2048
                }
            )
            
            elapsed = time.time() - start_time
            content = response['message']['content']
            
            # Check for channel presence
            has_analysis = '<analysis>' in content and '</analysis>' in content
            has_commentary = '<commentary>' in content and '</commentary>' in content
            has_final = '<final>' in content and '</final>' in content
            
            result = {
                'query': query,
                'has_analysis': has_analysis,
                'has_commentary': has_commentary,
                'has_final': has_final,
                'all_channels': has_analysis and has_commentary and has_final,
                'response_length': len(content),
                'time_seconds': round(elapsed, 2)
            }
            
            results.append(result)
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Response length: {len(content)} chars")
            print(f"  Channels detected:")
            print(f"    <analysis>:   {'✓' if has_analysis else '✗'}")
            print(f"    <commentary>: {'✓' if has_commentary else '✗'}")
            print(f"    <final>:      {'✓' if has_final else '✗'}")
            
            # Extract and show each channel if present
            if has_analysis:
                try:
                    start = content.index('<analysis>') + 10
                    end = content.index('</analysis>')
                    analysis_text = content[start:end].strip()
                    print(f"\n  Analysis preview:")
                    print(f"    {analysis_text[:200]}...")
                except:
                    pass
            
            if has_commentary:
                try:
                    start = content.index('<commentary>') + 12
                    end = content.index('</commentary>')
                    commentary_text = content[start:end].strip()
                    print(f"\n  Commentary preview:")
                    print(f"    {commentary_text[:200]}...")
                except:
                    pass
            
            if has_final:
                try:
                    start = content.index('<final>') + 7
                    end = content.index('</final>')
                    final_text = content[start:end].strip()
                    print(f"\n  Final answer:")
                    print(f"    {final_text[:300]}...")
                except:
                    pass
            
            if not (has_analysis or has_commentary or has_final):
                print(f"\n  No channels detected. Response preview:")
                print(f"    {content[:400]}...")
                
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'query': query, 'error': str(e)})
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in results if 'error' not in r]
    if successful_tests:
        total = len(successful_tests)
        with_analysis = sum(1 for r in successful_tests if r['has_analysis'])
        with_commentary = sum(1 for r in successful_tests if r['has_commentary'])
        with_final = sum(1 for r in successful_tests if r['has_final'])
        with_all = sum(1 for r in successful_tests if r['all_channels'])
        
        print(f"\nChannel usage across {total} tests:")
        print(f"  <analysis>:   {with_analysis}/{total} ({100*with_analysis/total:.0f}%)")
        print(f"  <commentary>: {with_commentary}/{total} ({100*with_commentary/total:.0f}%)")
        print(f"  <final>:      {with_final}/{total} ({100*with_final/total:.0f}%)")
        print(f"  All channels: {with_all}/{total} ({100*with_all/total:.0f}%)")
    
    # Save results
    with open('harmony_format_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to harmony_format_results.json")
    
    return results

if __name__ == "__main__":
    test_harmony_format()