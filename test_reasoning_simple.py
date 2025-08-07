#!/usr/bin/env python3
"""
Simple reasoning accuracy test
"""

import ollama
import time
import json

def test_reasoning_simple():
    """Simple test of reasoning accuracy at different levels"""
    
    model = 'gpt-oss:latest'
    
    print("="*60)
    print("SIMPLE REASONING ACCURACY TEST")
    print("="*60)
    
    # Simple math problem
    problem = "What is 15% of 240?"
    correct_answer = 36
    
    results = {}
    
    for effort in ['low', 'medium', 'high']:
        print(f"\n### Testing {effort.upper()} reasoning")
        print("-"*40)
        
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
                    'temperature': 0.1,  # Very low for consistency
                    'num_predict': 1024
                }
            )
            
            elapsed = time.time() - start_time
            content = response['message']['content']
            tokens = response.get('eval_count', 0) or 0  # Handle None
            
            is_correct = '36' in content
            
            results[effort] = {
                'tokens': tokens,
                'time': round(elapsed, 2),
                'correct': is_correct,
                'response': content[:200]
            }
            
            print(f"  Tokens: {tokens}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
            print(f"  Response preview: {content[:150]}...")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[effort] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    correct_count = sum(1 for r in results.values() if r.get('correct', False))
    total = len([r for r in results.values() if 'error' not in r])
    
    if total > 0:
        print(f"\nAccuracy: {correct_count}/{total} = {100*correct_count/total:.0f}%")
        
        print("\nToken comparison:")
        for effort in ['low', 'medium', 'high']:
            if effort in results and 'tokens' in results[effort]:
                r = results[effort]
                print(f"  {effort:8s}: {r['tokens']:4d} tokens, Correct: {'✓' if r['correct'] else '✗'}")
    
    # Save results
    with open('reasoning_simple_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to reasoning_simple_results.json")
    
    return results

if __name__ == "__main__":
    test_reasoning_simple()