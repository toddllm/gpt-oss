#!/usr/bin/env python3
"""
Focused test for variable-effort reasoning
"""

import ollama
import time
import json

def test_variable_effort():
    """Test reasoning at low/medium/high effort levels with a single problem"""
    
    model = 'gpt-oss:latest'
    
    # Single test problem
    problem = "If a train travels 120 miles in 2 hours, then slows to half its speed for the next 3 hours, how far did it travel in total?"
    
    print("="*60)
    print("TESTING VARIABLE-EFFORT REASONING")
    print("="*60)
    print(f"\nProblem: {problem}")
    print("-"*60)
    
    results = {}
    
    for effort in ['low', 'medium', 'high']:
        print(f"\n### Testing {effort.upper()} effort")
        
        system_prompt = f"Reasoning: {effort}"
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': problem}
                ],
                stream=False,
                options={
                    'temperature': 0.7,
                    'num_predict': 2048
                }
            )
            
            elapsed = time.time() - start_time
            content = response['message']['content']
            
            # Extract metrics
            eval_count = response.get('eval_count', 0)
            prompt_eval_count = response.get('prompt_eval_count', 0)
            total_duration = response.get('total_duration', 0) / 1e9  # Convert to seconds
            
            results[effort] = {
                'response_length': len(content),
                'tokens_generated': eval_count,
                'prompt_tokens': prompt_eval_count,
                'time_seconds': round(elapsed, 2),
                'model_time': round(total_duration, 2),
                'tokens_per_second': round(eval_count/elapsed, 2) if elapsed > 0 else 0
            }
            
            print(f"  Response tokens: {eval_count}")
            print(f"  Response time: {elapsed:.2f}s")
            print(f"  Speed: {results[effort]['tokens_per_second']} tok/s")
            print(f"  Response preview (first 300 chars):")
            print(f"  {content[:300]}...")
            
            # Check if answer is correct (should be 300 miles)
            if '300' in content:
                print(f"  ✓ Correct answer found!")
            else:
                print(f"  ✗ Answer not found or incorrect")
                
        except Exception as e:
            print(f"  Error: {e}")
            results[effort] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all('tokens_generated' in results[e] for e in ['low', 'medium', 'high']):
        # Token comparison
        print("\nToken Generation Comparison:")
        for effort in ['low', 'medium', 'high']:
            r = results[effort]
            print(f"  {effort:8s}: {r['tokens_generated']:5d} tokens in {r['time_seconds']:6.2f}s ({r['tokens_per_second']:5.1f} tok/s)")
        
        # Relative comparison
        low_tokens = results['low']['tokens_generated']
        med_tokens = results['medium']['tokens_generated']
        high_tokens = results['high']['tokens_generated']
        
        if low_tokens > 0:
            print(f"\nRelative token increase:")
            print(f"  Medium vs Low: {med_tokens/low_tokens:.2f}x")
            print(f"  High vs Low:   {high_tokens/low_tokens:.2f}x")
            print(f"  High vs Med:   {high_tokens/med_tokens:.2f}x")
    
    # Save results
    with open('variable_effort_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to variable_effort_results.json")

if __name__ == "__main__":
    test_variable_effort()