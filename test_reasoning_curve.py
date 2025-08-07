#!/usr/bin/env python3
"""
Test reasoning-length vs accuracy curve
"""

import ollama
import time
import json
import matplotlib.pyplot as plt

def test_reasoning_curve():
    """Test how reasoning length affects accuracy"""
    
    model = 'gpt-oss:latest'
    
    print("="*60)
    print("TESTING REASONING-LENGTH VS ACCURACY CURVE")
    print("="*60)
    
    # Math problems with known answers
    test_problems = [
        {
            'question': "What is 15% of 240?",
            'answer': 36,
            'check': lambda x: '36' in str(x)
        },
        {
            'question': "If you buy 3 apples at $1.20 each and 2 oranges at $0.80 each, what's the total cost?",
            'answer': 5.20,
            'check': lambda x: '5.20' in str(x) or '5.2' in str(x) or '$5.20' in str(x) or '5.2' in str(x)
        },
        {
            'question': "A rectangle has length 8 and width 5. What is its perimeter?",
            'answer': 26,
            'check': lambda x: '26' in str(x)
        },
        {
            'question': "If a car travels 180 miles using 6 gallons of gas, what is its fuel efficiency in miles per gallon?",
            'answer': 30,
            'check': lambda x: '30' in str(x)
        },
        {
            'question': "What is the sum of all integers from 1 to 10?",
            'answer': 55,
            'check': lambda x: '55' in str(x)
        }
    ]
    
    results = {'low': [], 'medium': [], 'high': []}
    effort_stats = {'low': {'correct': 0, 'total': 0, 'tokens': [], 'times': []},
                    'medium': {'correct': 0, 'total': 0, 'tokens': [], 'times': []},
                    'high': {'correct': 0, 'total': 0, 'tokens': [], 'times': []}}
    
    for problem_idx, problem in enumerate(test_problems, 1):
        print(f"\n### Problem {problem_idx}: {problem['question'][:60]}...")
        print(f"    Expected answer: {problem['answer']}")
        print("-"*60)
        
        for effort in ['low', 'medium', 'high']:
            system_prompt = f"Reasoning: {effort}"
            
            start_time = time.time()
            
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': problem['question']}
                    ],
                    stream=False,
                    options={
                        'temperature': 0.3,  # Lower temperature for more consistent answers
                        'num_predict': 2048
                    }
                )
                
                elapsed = time.time() - start_time
                content = response['message']['content']
                tokens = response.get('eval_count', 0)
                
                is_correct = problem['check'](content)
                
                result = {
                    'problem': problem['question'][:30],
                    'effort': effort,
                    'tokens': tokens,
                    'time': round(elapsed, 2),
                    'correct': is_correct,
                    'answer_preview': content[:100]
                }
                
                results[effort].append(result)
                effort_stats[effort]['total'] += 1
                effort_stats[effort]['tokens'].append(tokens)
                effort_stats[effort]['times'].append(elapsed)
                if is_correct:
                    effort_stats[effort]['correct'] += 1
                
                print(f"  {effort:8s}: {tokens:4d} tokens, {elapsed:5.2f}s, Correct: {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                print(f"  {effort:8s}: Error - {e}")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    accuracy_data = []
    token_data = []
    
    for effort in ['low', 'medium', 'high']:
        stats = effort_stats[effort]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            avg_tokens = sum(stats['tokens']) / len(stats['tokens']) if stats['tokens'] else 0
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            
            accuracy_data.append(accuracy)
            token_data.append(avg_tokens)
            
            print(f"\n{effort.upper()} Reasoning:")
            print(f"  Accuracy: {stats['correct']}/{stats['total']} = {accuracy:.1%}")
            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Avg time: {avg_time:.2f}s")
    
    # Create visualization if we have data
    if len(accuracy_data) == 3 and len(token_data) == 3:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accuracy vs Effort Level
            efforts = ['Low', 'Medium', 'High']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            ax1.bar(efforts, [a*100 for a in accuracy_data], color=colors)
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_xlabel('Reasoning Effort')
            ax1.set_title('Accuracy vs Reasoning Effort')
            ax1.set_ylim(0, 105)
            for i, v in enumerate(accuracy_data):
                ax1.text(i, v*100 + 2, f'{v:.0%}', ha='center')
            
            # Tokens vs Accuracy
            ax2.plot(token_data, [a*100 for a in accuracy_data], 'o-', color='#45B7D1', markersize=10, linewidth=2)
            ax2.set_xlabel('Average Tokens Generated')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Reasoning Length vs Accuracy')
            ax2.grid(True, alpha=0.3)
            
            for i, effort in enumerate(efforts):
                ax2.annotate(effort, (token_data[i], accuracy_data[i]*100), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plt.savefig('reasoning_curve.png', dpi=150, bbox_inches='tight')
            print("\nVisualization saved to reasoning_curve.png")
            
        except Exception as e:
            print(f"\nCould not create visualization: {e}")
    
    # Save detailed results
    with open('reasoning_curve_results.json', 'w') as f:
        json.dump({
            'results': results,
            'statistics': {
                effort: {
                    'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'avg_tokens': sum(stats['tokens']) / len(stats['tokens']) if stats['tokens'] else 0,
                    'avg_time': sum(stats['times']) / len(stats['times']) if stats['times'] else 0
                }
                for effort, stats in effort_stats.items()
            }
        }, f, indent=2)
    
    print("\nDetailed results saved to reasoning_curve_results.json")
    
    return results

if __name__ == "__main__":
    test_reasoning_curve()