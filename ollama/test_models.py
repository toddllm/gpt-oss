#!/usr/bin/env python3
"""
Ollama Model Testing and Benchmarking Script
Tests response quality, performance, and consistency
"""

import json
import time
import subprocess
import argparse
import statistics
from typing import List, Dict, Any
from datetime import datetime

class OllamaModelTester:
    def __init__(self, model_name: str = "gpt-oss:latest"):
        self.model_name = model_name
        self.results = []
        
    def run_prompt(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Run a single prompt and measure response time"""
        start_time = time.time()
        
        cmd = [
            "ollama", "run", 
            "--verbose",
            self.model_name,
            prompt
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "prompt": prompt,
                "response": result.stdout.strip(),
                "response_time": response_time,
                "temperature": temperature,
                "success": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {
                "prompt": prompt,
                "response": None,
                "response_time": 60.0,
                "temperature": temperature,
                "success": False,
                "error": "Timeout after 60 seconds"
            }
        except Exception as e:
            return {
                "prompt": prompt,
                "response": None,
                "response_time": 0,
                "temperature": temperature,
                "success": False,
                "error": str(e)
            }
    
    def test_basic_prompts(self) -> List[Dict[str, Any]]:
        """Test basic prompt types"""
        print(f"\n🧪 Testing basic prompts with {self.model_name}...")
        
        test_prompts = [
            # Math
            ("What is 25 * 4?", "math"),
            # Code generation
            ("Write a Python function to calculate factorial", "code"),
            # Reasoning
            ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "reasoning"),
            # Creative
            ("Write a haiku about artificial intelligence", "creative"),
            # Factual
            ("What is the capital of France?", "factual"),
            # Instruction following
            ("Count from 1 to 5", "instruction"),
        ]
        
        results = []
        for prompt, category in test_prompts:
            print(f"\n  Testing {category}: {prompt[:50]}...")
            result = self.run_prompt(prompt)
            result["category"] = category
            results.append(result)
            
            if result["success"]:
                print(f"  ✓ Response time: {result['response_time']:.2f}s")
                print(f"  Response preview: {result['response'][:100]}...")
            else:
                print(f"  ✗ Error: {result['error']}")
        
        return results
    
    def test_performance(self, num_runs: int = 5) -> Dict[str, Any]:
        """Test model performance with repeated queries"""
        print(f"\n⚡ Performance testing with {num_runs} runs...")
        
        prompt = "Generate a random number between 1 and 100"
        times = []
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end="")
            result = self.run_prompt(prompt)
            if result["success"]:
                times.append(result["response_time"])
                print(f" {result['response_time']:.2f}s")
            else:
                print(f" Failed")
        
        if times:
            return {
                "model": self.model_name,
                "num_runs": num_runs,
                "successful_runs": len(times),
                "avg_response_time": statistics.mean(times),
                "min_response_time": min(times),
                "max_response_time": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "median_time": statistics.median(times)
            }
        else:
            return {
                "model": self.model_name,
                "error": "All runs failed"
            }
    
    def test_context_handling(self) -> Dict[str, Any]:
        """Test model's ability to handle context"""
        print(f"\n📝 Testing context handling...")
        
        # Test with increasing context
        base_text = "The quick brown fox jumps over the lazy dog. " * 10
        prompts = [
            f"Summarize this text in one sentence: {base_text}",
            f"Summarize this text in one sentence: {base_text * 5}",
            f"Summarize this text in one sentence: {base_text * 10}",
        ]
        
        results = []
        for i, prompt in enumerate(prompts):
            context_size = len(prompt.split())
            print(f"  Testing with ~{context_size} words of context...")
            result = self.run_prompt(prompt)
            result["context_words"] = context_size
            results.append(result)
            
            if result["success"]:
                print(f"  ✓ Handled {context_size} words in {result['response_time']:.2f}s")
            else:
                print(f"  ✗ Failed with {context_size} words")
        
        return {
            "model": self.model_name,
            "context_tests": results
        }
    
    def test_consistency(self, prompt: str = "What is 2+2?", num_runs: int = 3) -> Dict[str, Any]:
        """Test response consistency with temperature=0"""
        print(f"\n🔄 Testing response consistency...")
        
        responses = []
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end="")
            # Run with temperature=0 for deterministic responses
            cmd = ["ollama", "run", "--parameter", "temperature", "0", self.model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                responses.append(response)
                print(" Done")
            else:
                print(" Failed")
        
        # Check if all responses are identical
        unique_responses = list(set(responses))
        consistency = len(unique_responses) == 1 if responses else False
        
        return {
            "model": self.model_name,
            "prompt": prompt,
            "num_runs": num_runs,
            "unique_responses": len(unique_responses),
            "is_consistent": consistency,
            "responses": unique_responses[:3]  # Show up to 3 unique responses
        }
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a formatted test report"""
        report = []
        report.append("=" * 60)
        report.append(f"OLLAMA MODEL TEST REPORT")
        report.append(f"Model: {self.model_name}")
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Basic tests summary
        if "basic_tests" in test_results:
            report.append("\n📊 BASIC TESTS SUMMARY:")
            success_count = sum(1 for t in test_results["basic_tests"] if t["success"])
            total_count = len(test_results["basic_tests"])
            report.append(f"  Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            
            for test in test_results["basic_tests"]:
                status = "✓" if test["success"] else "✗"
                report.append(f"  {status} {test['category']}: {test['response_time']:.2f}s")
        
        # Performance summary
        if "performance" in test_results:
            perf = test_results["performance"]
            report.append("\n⚡ PERFORMANCE METRICS:")
            if "error" not in perf:
                report.append(f"  Average Response Time: {perf['avg_response_time']:.2f}s")
                report.append(f"  Min/Max: {perf['min_response_time']:.2f}s / {perf['max_response_time']:.2f}s")
                report.append(f"  Median: {perf['median_time']:.2f}s")
                report.append(f"  Std Dev: {perf['std_dev']:.2f}s")
            else:
                report.append(f"  Error: {perf['error']}")
        
        # Consistency summary
        if "consistency" in test_results:
            cons = test_results["consistency"]
            report.append("\n🔄 CONSISTENCY CHECK:")
            report.append(f"  Consistent: {'Yes' if cons['is_consistent'] else 'No'}")
            report.append(f"  Unique Responses: {cons['unique_responses']}")
        
        # Context handling summary
        if "context" in test_results:
            ctx = test_results["context"]
            report.append("\n📝 CONTEXT HANDLING:")
            for test in ctx["context_tests"]:
                status = "✓" if test["success"] else "✗"
                report.append(f"  {status} {test['context_words']} words: {test['response_time']:.2f}s")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        print(f"\n🚀 Starting comprehensive model testing for {self.model_name}\n")
        
        all_results = {}
        
        # Run basic tests
        all_results["basic_tests"] = self.test_basic_prompts()
        
        # Run performance tests
        all_results["performance"] = self.test_performance(num_runs=5)
        
        # Run consistency tests
        all_results["consistency"] = self.test_consistency(num_runs=3)
        
        # Run context tests
        all_results["context"] = self.test_context_handling()
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Test Ollama models")
    parser.add_argument("--model", default="gpt-oss:latest", help="Model name to test")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    tester = OllamaModelTester(args.model)
    
    if args.quick:
        # Quick test - just basic prompts
        results = {"basic_tests": tester.test_basic_prompts()}
    else:
        # Full test suite
        results = tester.run_all_tests()
    
    # Generate and print report
    report = tester.generate_report(results)
    print("\n" + report)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()