#!/usr/bin/env python3
"""
Test hallucination baseline on factual questions
"""

import ollama
import time
import json

def test_hallucination():
    """Test hallucination rate on factual questions"""
    
    model = 'gpt-oss:latest'
    
    print("="*60)
    print("TESTING HALLUCINATION BASELINE")
    print("="*60)
    
    # Factual questions with known answers
    test_questions = [
        {
            'question': "What year did the Apollo 11 mission land on the moon?",
            'answer': "1969",
            'check': lambda x: '1969' in x
        },
        {
            'question': "What is the capital of Australia?",
            'answer': "Canberra",
            'check': lambda x: 'canberra' in x.lower()
        },
        {
            'question': "Who wrote the novel '1984'?",
            'answer': "George Orwell",
            'check': lambda x: 'orwell' in x.lower() or 'george orwell' in x.lower()
        },
        {
            'question': "What is the chemical symbol for gold?",
            'answer': "Au",
            'check': lambda x: 'Au' in x or ('gold' in x.lower() and 'au' in x.lower())
        },
        {
            'question': "How many continents are there?",
            'answer': "7",
            'check': lambda x: '7' in x or 'seven' in x.lower()
        },
        {
            'question': "What is the speed of light in vacuum (approximately)?",
            'answer': "299,792,458 m/s or 3×10^8 m/s",
            'check': lambda x: '299' in x or '3×10' in x or '3x10' in x.lower() or '300,000' in x or '186,000' in x
        },
        {
            'question': "Who painted the Mona Lisa?",
            'answer': "Leonardo da Vinci",
            'check': lambda x: 'leonardo' in x.lower() or 'da vinci' in x.lower()
        },
        {
            'question': "What is the largest planet in our solar system?",
            'answer': "Jupiter",
            'check': lambda x: 'jupiter' in x.lower()
        }
    ]
    
    results_by_effort = {}
    
    for effort in ['low', 'medium', 'high']:
        print(f"\n### Testing with {effort.upper()} reasoning")
        print("-"*40)
        
        results = []
        correct_count = 0
        
        for q in test_questions:
            start_time = time.time()
            
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': f'Reasoning: {effort}'},
                        {'role': 'user', 'content': q['question']}
                    ],
                    stream=False,
                    options={
                        'temperature': 0.3,  # Lower temperature for factual accuracy
                        'num_predict': 256
                    }
                )
                
                elapsed = time.time() - start_time
                content = response['message']['content']
                is_correct = q['check'](content)
                
                if is_correct:
                    correct_count += 1
                
                result = {
                    'question': q['question'],
                    'expected': q['answer'],
                    'correct': is_correct,
                    'time': round(elapsed, 2),
                    'response': content[:150]
                }
                
                results.append(result)
                
                print(f"  Q: {q['question'][:40]}...")
                print(f"    Expected: {q['answer']}")
                print(f"    Correct: {'✓' if is_correct else '✗'}")
                print(f"    Response: {content[:100]}...")
                
            except Exception as e:
                print(f"  Error with question: {e}")
                results.append({'question': q['question'], 'error': str(e)})
        
        # Calculate accuracy for this effort level
        total = len([r for r in results if 'error' not in r])
        accuracy = correct_count / total if total > 0 else 0
        
        results_by_effort[effort] = {
            'results': results,
            'correct': correct_count,
            'total': total,
            'accuracy': round(accuracy, 3)
        }
        
        print(f"\n  {effort.upper()} accuracy: {correct_count}/{total} = {accuracy:.1%}")
    
    # Overall summary
    print("\n" + "="*60)
    print("HALLUCINATION SUMMARY")
    print("="*60)
    
    print("\nAccuracy by reasoning level:")
    for effort in ['low', 'medium', 'high']:
        stats = results_by_effort[effort]
        print(f"  {effort:8s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    # Check if accuracy improves with higher reasoning
    accuracies = [results_by_effort[e]['accuracy'] for e in ['low', 'medium', 'high']]
    if accuracies[2] > accuracies[0]:
        improvement = (accuracies[2] - accuracies[0]) * 100
        print(f"\nImprovement from low to high: +{improvement:.1f} percentage points")
    
    # Per-question analysis
    print("\nPer-question accuracy across all levels:")
    question_accuracy = {}
    for i, q in enumerate(test_questions):
        question_text = q['question'][:40] + "..."
        correct_across_levels = 0
        for effort in ['low', 'medium', 'high']:
            if i < len(results_by_effort[effort]['results']):
                result = results_by_effort[effort]['results'][i]
                if 'error' not in result and result['correct']:
                    correct_across_levels += 1
        
        accuracy_pct = (correct_across_levels / 3) * 100
        question_accuracy[question_text] = accuracy_pct
        print(f"  {question_text:45s}: {accuracy_pct:.0f}%")
    
    # Identify problematic questions
    problematic = [q for q, acc in question_accuracy.items() if acc < 50]
    if problematic:
        print(f"\nProblematic questions (< 50% accuracy):")
        for q in problematic:
            print(f"  - {q}")
    
    # Save results
    with open('hallucination_results.json', 'w') as f:
        json.dump(results_by_effort, f, indent=2)
    
    print("\nResults saved to hallucination_results.json")
    
    return results_by_effort

if __name__ == "__main__":
    test_hallucination()