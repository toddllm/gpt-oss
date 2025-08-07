#!/usr/bin/env python3
"""
Test long context handling
"""

import ollama
import time
import json

def test_long_context():
    """Test long context handling capability"""
    
    model = 'gpt-oss:latest'
    
    print("="*60)
    print("TESTING LONG CONTEXT HANDLING")
    print("="*60)
    
    # Create a long context with a hidden fact
    intro = "This is a long document about various topics. Please read carefully.\n\n"
    
    # Add filler content
    filler_sections = []
    for i in range(1, 20):
        filler_sections.append(f"""
Chapter {i}: Topic {i}
This is chapter {i} discussing various aspects of topic {i}. 
The content here is meant to fill space and create a longer context.
Important facts about topic {i} include various details and information.
This chapter contains approximately 100 words of content to make the document longer.
We discuss multiple aspects and provide comprehensive coverage of the subject matter.
Additional details are provided to ensure thorough understanding of the concepts.
The chapter concludes with a summary of the key points discussed above.
""")
    
    # Add the secret information near the end
    secret_section = """
Chapter 42: Special Information
Hidden within this chapter is a crucial piece of information.
The secret code for this document is: ALPHA-BRAVO-2024.
This code should be remembered as it will be important later.
Please note this code carefully: ALPHA-BRAVO-2024.
"""
    
    # Add more filler after the secret
    more_filler = """
Chapter 50: Final Thoughts
This document has covered many topics and provided extensive information.
We hope you have found the content useful and informative.
Thank you for reading through this entire document.
"""
    
    # Combine all sections
    long_text = intro + "\n\n".join(filler_sections) + "\n\n" + secret_section + "\n\n" + more_filler
    
    context_chars = len(long_text)
    context_words = len(long_text.split())
    
    print(f"Context size: {context_chars:,} characters ({context_words:,} words)")
    
    # Test questions
    test_questions = [
        {
            'question': "What is the secret code mentioned in Chapter 42?",
            'expected': "ALPHA-BRAVO-2024",
            'check': lambda x: "ALPHA-BRAVO-2024" in x or "ALPHA" in x and "BRAVO" in x and "2024" in x
        },
        {
            'question': "What chapter contains the secret code?",
            'expected': "42",
            'check': lambda x: "42" in x or "forty-two" in x.lower()
        }
    ]
    
    results = []
    
    for test in test_questions:
        print(f"\n### Question: {test['question']}")
        print("-"*40)
        
        full_prompt = f"{long_text}\n\nQuestion: {test['question']}"
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': 'Reasoning: high'},
                    {'role': 'user', 'content': full_prompt}
                ],
                stream=False,
                options={
                    'temperature': 0.3,
                    'num_predict': 512,
                    'num_ctx': 8192  # Ensure we have enough context window
                }
            )
            
            elapsed = time.time() - start_time
            content = response['message']['content']
            
            is_correct = test['check'](content)
            
            result = {
                'question': test['question'],
                'expected': test['expected'],
                'correct': is_correct,
                'time_seconds': round(elapsed, 2),
                'response_length': len(content),
                'response_preview': content[:200]
            }
            
            results.append(result)
            
            print(f"  Expected: {test['expected']}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
            print(f"  Response: {content[:200]}...")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'question': test['question'], 'error': str(e)})
    
    # Test with different context sizes
    print("\n" + "="*60)
    print("CONTEXT SIZE SCALING TEST")
    print("="*60)
    
    context_sizes = [1000, 5000, 10000]
    scaling_results = []
    
    for size in context_sizes:
        # Create context of specific size
        test_text = "This is a test. " * (size // 16)  # Approximately size chars
        test_text = test_text[:size-100] + " The magic number is 42. "
        
        question = "What is the magic number?"
        
        print(f"\nTesting with ~{size} character context")
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': 'Reasoning: medium'},
                    {'role': 'user', 'content': f"{test_text}\n\n{question}"}
                ],
                stream=False,
                options={
                    'temperature': 0.3,
                    'num_predict': 256
                }
            )
            
            elapsed = time.time() - start_time
            content = response['message']['content']
            found = "42" in content
            
            scaling_results.append({
                'context_size': size,
                'time': round(elapsed, 2),
                'found_answer': found
            })
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Found answer: {'✓' if found else '✗'}")
            
        except Exception as e:
            print(f"  Error: {e}")
            scaling_results.append({'context_size': size, 'error': str(e)})
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in results if 'error' not in r]
    if successful:
        correct = sum(1 for r in successful if r['correct'])
        print(f"\nMain test accuracy: {correct}/{len(successful)} = {100*correct/len(successful):.0f}%")
    
    successful_scaling = [r for r in scaling_results if 'error' not in r and r.get('found_answer')]
    if successful_scaling:
        print(f"\nContext scaling success: {len(successful_scaling)}/{len(context_sizes)}")
        print("Processing times by context size:")
        for r in scaling_results:
            if 'error' not in r:
                print(f"  {r['context_size']:5d} chars: {r['time']:5.2f}s")
    
    # Save results
    all_results = {
        'main_test': {
            'context_size': context_chars,
            'results': results
        },
        'scaling_test': scaling_results
    }
    
    with open('long_context_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to long_context_results.json")
    
    return all_results

if __name__ == "__main__":
    test_long_context()