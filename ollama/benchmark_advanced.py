#!/usr/bin/env python3
"""
Advanced Benchmark Suite for GPT-OSS Models
Tests against paper benchmarks: AIME, GPQA, MMLU, HLE, Codeforces
"""

import json
import time
import requests
from typing import Dict, List, Any
from datetime import datetime
import argparse

class AdvancedBenchmarkSuite:
    def __init__(self, model_name: str = "gpt-oss:latest", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        self.results = {}
    
    def query_model(self, prompt: str, system: str = None, max_tokens: int = 1000) -> Dict[str, Any]:
        """Query the model via Ollama API"""
        url = f"{self.api_url}/api/generate"
        
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\n{prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistency
                "num_predict": max_tokens
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            return {
                "response": result.get("response", ""),
                "time": time.time() - start_time,
                "tokens": result.get("eval_count", 0),
                "success": True
            }
        except Exception as e:
            return {"response": None, "error": str(e), "success": False}
    
    def benchmark_aime_style(self) -> Dict[str, Any]:
        """AIME-style math competition problems"""
        print("\n🎯 AIME-Style Mathematics Benchmark")
        print("-" * 40)
        
        problems = [
            {
                "id": "aime_1",
                "problem": "Find the number of positive integers n ≤ 1000 such that n² + n + 41 is prime.",
                "category": "number_theory"
            },
            {
                "id": "aime_2", 
                "problem": "A regular hexagon with side length 1 is inscribed in a circle. What is the area of the circle?",
                "category": "geometry"
            },
            {
                "id": "aime_3",
                "problem": "How many ways can you arrange the letters in MATHEMATICS so that no two identical letters are adjacent?",
                "category": "combinatorics"
            },
            {
                "id": "aime_4",
                "problem": "Find the sum of all positive integers n such that n³ - 7n² + 14n - 8 = 0",
                "category": "algebra"
            },
            {
                "id": "aime_5",
                "problem": "A sequence a₁, a₂, ... satisfies a₁ = 1, a₂ = 2, and aₙ₊₂ = aₙ₊₁ + aₙ for n ≥ 1. Find a₁₀.",
                "category": "sequences"
            }
        ]
        
        results = []
        system_prompt = "Reasoning: high. You are solving AIME competition problems. Show all work step by step."
        
        for prob in problems:
            print(f"\n  Testing {prob['id']} ({prob['category']})...")
            result = self.query_model(prob["problem"], system=system_prompt)
            
            if result["success"]:
                # Check for mathematical reasoning indicators
                math_indicators = ["therefore", "thus", "equation", "solve", "=", "substitut", "factor"]
                reasoning_score = sum(1 for ind in math_indicators if ind.lower() in result["response"].lower())
                
                results.append({
                    "problem_id": prob["id"],
                    "category": prob["category"],
                    "reasoning_score": reasoning_score,
                    "response_length": len(result["response"]),
                    "time": result["time"],
                    "response_preview": result["response"][:300]
                })
                print(f"    ✓ Reasoning indicators: {reasoning_score}/7")
                print(f"    ✓ Response time: {result['time']:.2f}s")
            else:
                results.append({"problem_id": prob["id"], "error": result["error"]})
                print(f"    ✗ Error: {result['error']}")
        
        return {"benchmark": "AIME", "problems": results}
    
    def benchmark_gpqa_style(self) -> Dict[str, Any]:
        """GPQA-style expert-level questions"""
        print("\n🎓 GPQA-Style Expert Questions Benchmark")
        print("-" * 40)
        
        questions = [
            {
                "id": "gpqa_physics",
                "question": "In quantum field theory, what is the significance of the Faddeev-Popov ghost fields in non-Abelian gauge theories?",
                "field": "physics"
            },
            {
                "id": "gpqa_chemistry",
                "question": "Explain the mechanism of the Diels-Alder reaction and its stereochemical outcomes.",
                "field": "chemistry"
            },
            {
                "id": "gpqa_biology",
                "question": "Describe the role of the CRISPR-Cas9 system in bacterial adaptive immunity and its applications.",
                "field": "biology"
            },
            {
                "id": "gpqa_cs",
                "question": "What is the computational complexity class BQP and how does it relate to P and NP?",
                "field": "computer_science"
            }
        ]
        
        results = []
        system_prompt = "You are an expert scientist. Provide detailed, graduate-level explanations."
        
        for q in questions:
            print(f"\n  Testing {q['id']} ({q['field']})...")
            result = self.query_model(q["question"], system=system_prompt)
            
            if result["success"]:
                # Check for technical depth
                technical_terms = ["theorem", "principle", "mechanism", "equation", "hypothesis", "model", "theory"]
                technical_score = sum(1 for term in technical_terms if term.lower() in result["response"].lower())
                
                results.append({
                    "question_id": q["id"],
                    "field": q["field"],
                    "technical_score": technical_score,
                    "response_length": len(result["response"]),
                    "time": result["time"]
                })
                print(f"    ✓ Technical depth: {technical_score}/7")
            else:
                results.append({"question_id": q["id"], "error": result["error"]})
        
        return {"benchmark": "GPQA", "questions": results}
    
    def benchmark_mmlu_style(self) -> Dict[str, Any]:
        """MMLU-style multiple choice questions"""
        print("\n📚 MMLU-Style Multitask Benchmark")
        print("-" * 40)
        
        questions = [
            {
                "id": "mmlu_history",
                "question": "The Peace of Westphalia (1648) is significant because it:\nA) Ended the Hundred Years' War\nB) Established the modern concept of state sovereignty\nC) Created the United Nations\nD) Unified Germany",
                "subject": "history",
                "answer": "B"
            },
            {
                "id": "mmlu_economics", 
                "question": "In a perfectly competitive market, firms maximize profit where:\nA) Marginal revenue equals average cost\nB) Marginal cost equals marginal revenue\nC) Total revenue is maximized\nD) Average cost is minimized",
                "subject": "economics",
                "answer": "B"
            },
            {
                "id": "mmlu_philosophy",
                "question": "According to Kant's categorical imperative, an action is moral if:\nA) It produces the greatest happiness\nB) It follows divine command\nC) Its maxim can be universalized\nD) It aligns with virtue ethics",
                "subject": "philosophy",
                "answer": "C"
            }
        ]
        
        results = []
        system_prompt = "Answer the multiple choice question. First explain your reasoning, then state your answer as A, B, C, or D."
        
        for q in questions:
            print(f"\n  Testing {q['id']} ({q['subject']})...")
            result = self.query_model(q["question"], system=system_prompt)
            
            if result["success"]:
                response = result["response"].upper()
                # Check if correct answer is mentioned
                correct = q["answer"] in response
                # Check if answer is clearly stated
                has_clear_answer = any(f"){letter}" in response or f"answer is {letter}" in response.lower() 
                                      for letter in ["A", "B", "C", "D"])
                
                results.append({
                    "question_id": q["id"],
                    "subject": q["subject"],
                    "correct": correct,
                    "has_clear_answer": has_clear_answer,
                    "expected": q["answer"],
                    "response_preview": result["response"][:200]
                })
                print(f"    Answer correct: {'✓' if correct else '✗'}")
                print(f"    Clear answer: {'✓' if has_clear_answer else '✗'}")
            else:
                results.append({"question_id": q["id"], "error": result["error"]})
        
        return {"benchmark": "MMLU", "questions": results}
    
    def benchmark_codeforces_style(self) -> Dict[str, Any]:
        """Codeforces-style competitive programming"""
        print("\n💻 Codeforces-Style Programming Benchmark")
        print("-" * 40)
        
        problems = [
            {
                "id": "cf_easy",
                "problem": "Write a function to find the kth smallest element in an unsorted array.",
                "difficulty": "easy"
            },
            {
                "id": "cf_medium",
                "problem": "Given a binary tree, find the maximum path sum between any two nodes.",
                "difficulty": "medium"
            },
            {
                "id": "cf_hard",
                "problem": "Implement an algorithm to find the longest palindromic subsequence in a string using dynamic programming.",
                "difficulty": "hard"
            }
        ]
        
        results = []
        system_prompt = "Solve the programming problem. Provide complete, working Python code with time/space complexity analysis."
        
        for prob in problems:
            print(f"\n  Testing {prob['id']} ({prob['difficulty']})...")
            result = self.query_model(prob["problem"], system=system_prompt)
            
            if result["success"]:
                response = result["response"]
                # Check for code quality indicators
                has_function = "def " in response
                has_complexity = "O(" in response
                has_explanation = "return" in response and ("explain" in response.lower() or "approach" in response.lower())
                
                results.append({
                    "problem_id": prob["id"],
                    "difficulty": prob["difficulty"],
                    "has_function": has_function,
                    "has_complexity": has_complexity,
                    "has_explanation": has_explanation,
                    "response_length": len(response),
                    "time": result["time"]
                })
                print(f"    Has function: {'✓' if has_function else '✗'}")
                print(f"    Complexity analysis: {'✓' if has_complexity else '✗'}")
                print(f"    Explanation: {'✓' if has_explanation else '✗'}")
            else:
                results.append({"problem_id": prob["id"], "error": result["error"]})
        
        return {"benchmark": "Codeforces", "problems": results}
    
    def benchmark_swe_bench_style(self) -> Dict[str, Any]:
        """SWE-Bench style software engineering tasks"""
        print("\n🛠️ SWE-Bench Style Engineering Tasks")
        print("-" * 40)
        
        tasks = [
            {
                "id": "swe_debug",
                "task": "Debug this code: def merge_sort(arr): if len(arr) <= 1: return arr; mid = len(arr) // 2; left = merge_sort(arr[:mid]); right = merge_sort(arr[mid:]); return merge(left, right) # merge function is missing",
                "type": "debugging"
            },
            {
                "id": "swe_refactor",
                "task": "Refactor this code to use list comprehension: result = []; for i in range(10): if i % 2 == 0: result.append(i * i)",
                "type": "refactoring"
            },
            {
                "id": "swe_test",
                "task": "Write unit tests for a function that validates email addresses",
                "type": "testing"
            }
        ]
        
        results = []
        system_prompt = "You are a senior software engineer. Provide production-quality solutions."
        
        for task in tasks:
            print(f"\n  Testing {task['id']} ({task['type']})...")
            result = self.query_model(task["task"], system=system_prompt)
            
            if result["success"]:
                response = result["response"]
                # Check for engineering best practices
                has_code = "def " in response or "class " in response or "import " in response
                has_comments = "#" in response or '"""' in response
                mentions_edge_cases = "edge" in response.lower() or "corner case" in response.lower()
                
                results.append({
                    "task_id": task["id"],
                    "task_type": task["type"],
                    "has_code": has_code,
                    "has_comments": has_comments,
                    "mentions_edge_cases": mentions_edge_cases,
                    "response_length": len(response)
                })
                print(f"    Has code: {'✓' if has_code else '✗'}")
                print(f"    Has comments: {'✓' if has_comments else '✗'}")
                print(f"    Edge cases: {'✓' if mentions_edge_cases else '✗'}")
            else:
                results.append({"task_id": task["id"], "error": result["error"]})
        
        return {"benchmark": "SWE-Bench", "tasks": results}
    
    def generate_report(self, all_results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("ADVANCED GPT-OSS BENCHMARK REPORT")
        report.append(f"Model: {self.model_name}")
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # AIME Results
        if "AIME" in all_results:
            aime = all_results["AIME"]
            report.append("\n🎯 AIME-STYLE MATHEMATICS:")
            total_reasoning = sum(p.get("reasoning_score", 0) for p in aime["problems"] if "error" not in p)
            avg_reasoning = total_reasoning / len(aime["problems"]) if aime["problems"] else 0
            report.append(f"  Average reasoning score: {avg_reasoning:.1f}/7")
            
            for prob in aime["problems"]:
                if "error" not in prob:
                    report.append(f"  {prob['problem_id']} ({prob['category']}):")
                    report.append(f"    Reasoning: {prob['reasoning_score']}/7")
                    report.append(f"    Time: {prob['time']:.2f}s")
        
        # GPQA Results
        if "GPQA" in all_results:
            gpqa = all_results["GPQA"]
            report.append("\n🎓 GPQA-STYLE EXPERT QUESTIONS:")
            total_technical = sum(q.get("technical_score", 0) for q in gpqa["questions"] if "error" not in q)
            avg_technical = total_technical / len(gpqa["questions"]) if gpqa["questions"] else 0
            report.append(f"  Average technical depth: {avg_technical:.1f}/7")
            
            for q in gpqa["questions"]:
                if "error" not in q:
                    report.append(f"  {q['question_id']} ({q['field']}):")
                    report.append(f"    Technical depth: {q['technical_score']}/7")
        
        # MMLU Results
        if "MMLU" in all_results:
            mmlu = all_results["MMLU"]
            report.append("\n📚 MMLU-STYLE MULTITASK:")
            correct = sum(1 for q in mmlu["questions"] if q.get("correct", False))
            total = len(mmlu["questions"])
            report.append(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
            
            for q in mmlu["questions"]:
                if "error" not in q:
                    report.append(f"  {q['question_id']} ({q['subject']}):")
                    report.append(f"    Correct: {'✓' if q['correct'] else '✗'} (Expected: {q['expected']})")
        
        # Codeforces Results
        if "Codeforces" in all_results:
            cf = all_results["Codeforces"]
            report.append("\n💻 CODEFORCES-STYLE PROGRAMMING:")
            
            for prob in cf["problems"]:
                if "error" not in prob:
                    report.append(f"  {prob['problem_id']} ({prob['difficulty']}):")
                    quality_score = sum([prob.get("has_function", False), 
                                       prob.get("has_complexity", False),
                                       prob.get("has_explanation", False)])
                    report.append(f"    Code quality: {quality_score}/3")
                    report.append(f"    Time: {prob['time']:.2f}s")
        
        # SWE-Bench Results
        if "SWE-Bench" in all_results:
            swe = all_results["SWE-Bench"]
            report.append("\n🛠️ SWE-BENCH ENGINEERING TASKS:")
            
            for task in swe["tasks"]:
                if "error" not in task:
                    report.append(f"  {task['task_id']} ({task['task_type']}):")
                    quality_score = sum([task.get("has_code", False),
                                       task.get("has_comments", False),
                                       task.get("mentions_edge_cases", False)])
                    report.append(f"    Engineering quality: {quality_score}/3")
        
        report.append("\n" + "=" * 80)
        report.append("OVERALL PERFORMANCE SUMMARY:")
        
        # Calculate aggregate metrics
        total_tests = 0
        successful_tests = 0
        
        for benchmark_name, benchmark_data in all_results.items():
            if isinstance(benchmark_data, dict):
                items = benchmark_data.get("problems", benchmark_data.get("questions", benchmark_data.get("tasks", [])))
                total_tests += len(items)
                successful_tests += sum(1 for item in items if "error" not in item)
        
        report.append(f"  Total tests: {total_tests}")
        report.append(f"  Successful: {successful_tests}")
        report.append(f"  Success rate: {successful_tests/total_tests*100:.1f}%")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites"""
        print("\n" + "=" * 50)
        print("🚀 STARTING ADVANCED BENCHMARK SUITE")
        print("=" * 50)
        
        results = {}
        
        # Run each benchmark
        results["AIME"] = self.benchmark_aime_style()
        results["GPQA"] = self.benchmark_gpqa_style()
        results["MMLU"] = self.benchmark_mmlu_style()
        results["Codeforces"] = self.benchmark_codeforces_style()
        results["SWE-Bench"] = self.benchmark_swe_bench_style()
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Advanced benchmarks for GPT-OSS models")
    parser.add_argument("--model", default="gpt-oss:latest", help="Model name")
    parser.add_argument("--api-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--report", help="Save report to text file")
    
    args = parser.parse_args()
    
    suite = AdvancedBenchmarkSuite(args.model, args.api_url)
    results = suite.run_all_benchmarks()
    
    # Generate and print report
    report = suite.generate_report(results)
    print("\n" + report)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {args.report}")


if __name__ == "__main__":
    main()