#!/usr/bin/env python3
"""
Tool-Enhanced Agent Framework for GPT-OSS Models
Implements agentic workflows with real and simulated tools
"""

import json
import time
import subprocess
import requests
import ast
import re
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod

class Tool(ABC):
    """Base class for agent tools"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        pass

class PythonTool(Tool):
    """Execute Python code in isolated environment"""
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            # Create safe execution environment
            safe_globals = {
                "__builtins__": {
                    "len": len, "range": range, "print": print,
                    "int": int, "float": float, "str": str,
                    "list": list, "dict": dict, "tuple": tuple,
                    "sum": sum, "min": min, "max": max,
                    "sorted": sorted, "reversed": reversed,
                    "enumerate": enumerate, "zip": zip,
                    "map": map, "filter": filter,
                    "abs": abs, "round": round,
                    "all": all, "any": any,
                }
            }
            
            # Import common libraries
            exec("import math", safe_globals)
            exec("import statistics", safe_globals)
            exec("import itertools", safe_globals)
            exec("import collections", safe_globals)
            
            # Capture output
            output = []
            safe_globals["print"] = lambda *args: output.append(" ".join(map(str, args)))
            
            # Execute code
            exec(code, safe_globals)
            
            return {
                "success": True,
                "output": "\n".join(output) if output else "Code executed successfully",
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "python",
            "description": "Execute Python code",
            "parameters": {
                "code": {"type": "string", "description": "Python code to execute"}
            }
        }

class WebSearchTool(Tool):
    """Simulated web search tool"""
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Simulate web search results"""
        # In production, would use real search API
        simulated_results = {
            "gpt-oss": [
                {"title": "OpenAI GPT-OSS Released", "url": "openai.com/gpt-oss", 
                 "snippet": "GPT-OSS-120b and 20b models now available..."},
                {"title": "Benchmarks Show Strong Performance", "url": "example.com/benchmarks",
                 "snippet": "AIME: 96.6%, MMLU: 90%, Codeforces: 2620 Elo..."}
            ],
            "fine-tuning": [
                {"title": "Fine-Tuning Guide", "url": "huggingface.co/docs/fine-tune",
                 "snippet": "Use LoRA for efficient fine-tuning on consumer GPUs..."},
                {"title": "Unsloth Optimization", "url": "github.com/unsloth",
                 "snippet": "2x faster training, 70% less VRAM..."}
            ]
        }
        
        # Find relevant results
        results = []
        for keyword, items in simulated_results.items():
            if keyword.lower() in query.lower():
                results.extend(items)
        
        if not results:
            results = [{"title": f"Search results for: {query}", 
                       "snippet": "No specific results in simulation"}]
        
        return {
            "success": True,
            "results": results[:3],
            "query": query
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            }
        }

class FileSystemTool(Tool):
    """Read/write files safely"""
    
    def __init__(self, base_path: str = "/home/tdeshane/gpt-oss/ollama"):
        self.base_path = base_path
    
    def execute(self, action: str, path: str, content: str = None) -> Dict[str, Any]:
        """File operations"""
        import os
        
        # Ensure path is within base directory
        full_path = os.path.join(self.base_path, path)
        if not full_path.startswith(self.base_path):
            return {"success": False, "error": "Path outside allowed directory"}
        
        try:
            if action == "read":
                with open(full_path, 'r') as f:
                    content = f.read()
                return {"success": True, "content": content[:1000]}  # Limit size
            
            elif action == "write":
                if content:
                    with open(full_path, 'w') as f:
                        f.write(content)
                    return {"success": True, "message": f"Written to {path}"}
                return {"success": False, "error": "No content provided"}
            
            elif action == "list":
                items = os.listdir(full_path if os.path.isdir(full_path) else self.base_path)
                return {"success": True, "items": items[:20]}  # Limit items
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "file_system",
            "description": "File system operations",
            "parameters": {
                "action": {"type": "string", "enum": ["read", "write", "list"]},
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content for write", "optional": True}
            }
        }

class AnalysisTool(Tool):
    """Analyze code or text for patterns"""
    
    def execute(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze text/code"""
        results = {}
        
        if analysis_type in ["code", "general"]:
            # Code quality metrics
            results["metrics"] = {
                "lines": len(text.split('\n')),
                "words": len(text.split()),
                "functions": text.count("def "),
                "classes": text.count("class "),
                "imports": text.count("import "),
                "comments": text.count("#") + text.count('"""'),
            }
            
            # Complexity indicators
            complexity_indicators = ["for", "while", "if", "try", "except"]
            results["complexity"] = sum(text.count(ind) for ind in complexity_indicators)
        
        if analysis_type in ["security", "general"]:
            # Security checks
            security_risks = {
                "eval": text.count("eval("),
                "exec": text.count("exec("),
                "os_system": text.count("os.system"),
                "__import__": text.count("__import__"),
                "pickle": text.count("pickle."),
            }
            results["security_risks"] = {k: v for k, v in security_risks.items() if v > 0}
        
        if analysis_type in ["quality", "general"]:
            # Quality indicators
            results["quality"] = {
                "has_docstrings": '"""' in text or "'''" in text,
                "has_type_hints": "->" in text or ": str" in text or ": int" in text,
                "has_tests": "test_" in text or "assert" in text,
                "follows_pep8": len([l for l in text.split('\n') if len(l) > 79]) < 5
            }
        
        return {
            "success": True,
            "analysis_type": analysis_type,
            "results": results
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "analyze",
            "description": "Analyze code or text",
            "parameters": {
                "text": {"type": "string", "description": "Text to analyze"},
                "analysis_type": {"type": "string", "enum": ["code", "security", "quality", "general"]}
            }
        }

class GPTOSSAgent:
    """Autonomous agent using GPT-OSS with tools"""
    
    def __init__(self, model_name: str = "gpt-oss:latest", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        self.tools = {}
        self.conversation_history = []
        self.max_iterations = 10
        
        # Register default tools
        self.register_tool(PythonTool())
        self.register_tool(WebSearchTool())
        self.register_tool(FileSystemTool())
        self.register_tool(AnalysisTool())
    
    def register_tool(self, tool: Tool):
        """Register a tool for the agent"""
        schema = tool.get_schema()
        self.tools[schema["name"]] = {
            "tool": tool,
            "schema": schema
        }
    
    def query_model(self, prompt: str, system: str = None) -> str:
        """Query the model"""
        url = f"{self.api_url}/api/generate"
        
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\n{prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1000
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error querying model: {str(e)}"
    
    def parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response for thoughts, actions, and results"""
        parsed = {
            "thought": None,
            "action": None,
            "tool_call": None,
            "final_answer": None
        }
        
        # Extract thought
        thought_match = re.search(r'(?:THOUGHT|THINK|Thinking):\s*(.+?)(?:ACTION|TOOL|$)', 
                                 response, re.IGNORECASE | re.DOTALL)
        if thought_match:
            parsed["thought"] = thought_match.group(1).strip()
        
        # Extract action/tool call
        action_match = re.search(r'(?:ACTION|TOOL|CALL):\s*(.+?)(?:RESULT|OBSERVATION|$)', 
                                response, re.IGNORECASE | re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
            
            # Try to parse as JSON
            json_match = re.search(r'\{.*\}', action_text, re.DOTALL)
            if json_match:
                try:
                    parsed["tool_call"] = json.loads(json_match.group())
                except:
                    # Try to parse as function call
                    func_match = re.search(r'(\w+)\((.*?)\)', action_text)
                    if func_match:
                        parsed["tool_call"] = {
                            "name": func_match.group(1),
                            "args": func_match.group(2)
                        }
        
        # Extract final answer
        answer_match = re.search(r'(?:FINAL|ANSWER|RESULT):\s*(.+?)$', 
                               response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            parsed["final_answer"] = answer_match.group(1).strip()
        
        return parsed
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]["tool"]
        try:
            result = tool.execute(**args)
            return result
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def run(self, task: str, max_iterations: int = None) -> Dict[str, Any]:
        """Run agent on a task"""
        if max_iterations:
            self.max_iterations = max_iterations
        
        print(f"\n🤖 Agent starting task: {task}")
        print("="*50)
        
        # Create system prompt with tool descriptions
        tools_desc = "\n".join([
            f"- {name}: {info['schema']['description']}"
            for name, info in self.tools.items()
        ])
        
        system_prompt = f"""You are an autonomous agent with access to tools.
Available tools:
{tools_desc}

Format your responses as:
THOUGHT: Your reasoning
ACTION: Tool call as JSON {{"name": "tool_name", "args": {{...}}}}
OBSERVATION: [Will be filled by system]
... (repeat as needed)
FINAL ANSWER: Your conclusion

Task: {task}"""
        
        conversation = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n📍 Iteration {iteration}")
            
            # Build context
            context = system_prompt
            if conversation:
                context += "\n\nPrevious steps:\n" + "\n".join(conversation[-3:])  # Last 3 exchanges
            
            # Get agent response
            response = self.query_model(context)
            print(f"Agent: {response[:200]}...")
            
            # Parse response
            parsed = self.parse_agent_response(response)
            
            if parsed["thought"]:
                print(f"💭 Thought: {parsed['thought'][:100]}...")
                conversation.append(f"THOUGHT: {parsed['thought']}")
            
            if parsed["tool_call"]:
                tool_name = parsed["tool_call"].get("name")
                tool_args = parsed["tool_call"].get("args", {})
                
                # Convert string args to dict if needed
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except:
                        # Try to parse as key=value pairs
                        tool_args = dict(re.findall(r'(\w+)=(["\']?)(.+?)\2', tool_args))
                
                print(f"🔧 Tool call: {tool_name}({tool_args})")
                
                # Execute tool
                result = self.execute_tool(tool_name, tool_args)
                print(f"📊 Result: {str(result)[:100]}...")
                
                conversation.append(f"ACTION: {tool_name}({tool_args})")
                conversation.append(f"OBSERVATION: {result}")
            
            if parsed["final_answer"]:
                print(f"\n✅ Final Answer: {parsed['final_answer']}")
                return {
                    "success": True,
                    "answer": parsed["final_answer"],
                    "iterations": iteration,
                    "conversation": conversation
                }
            
            # Check if stuck
            if iteration > 3 and not parsed["tool_call"]:
                print("⚠️ Agent seems stuck, forcing conclusion...")
                break
        
        # Fallback
        return {
            "success": False,
            "error": "Max iterations reached",
            "iterations": iteration,
            "conversation": conversation
        }
    
    def self_improve(self, task: str, success_criteria: Callable) -> Dict[str, Any]:
        """Self-improving loop: agent critiques and improves its own solution"""
        print(f"\n🔄 Self-Improving Agent: {task}")
        print("="*50)
        
        best_solution = None
        best_score = 0
        improvements = []
        
        for round in range(3):  # 3 improvement rounds
            print(f"\n🔄 Round {round + 1}")
            
            # Generate solution
            if round == 0:
                result = self.run(task)
            else:
                # Improve based on critique
                improvement_task = f"""Previous attempt:
{best_solution}

Critique: {improvements[-1] if improvements else 'N/A'}

Task: {task}
Improve the solution based on the critique."""
                result = self.run(improvement_task)
            
            if result.get("success"):
                solution = result.get("answer", "")
                
                # Evaluate solution
                score = success_criteria(solution)
                print(f"📊 Score: {score}")
                
                if score > best_score:
                    best_solution = solution
                    best_score = score
                
                # Self-critique
                critique_prompt = f"""Analyze this solution and suggest improvements:
Solution: {solution}
Task: {task}

Provide specific, actionable improvements."""
                
                critique = self.query_model(critique_prompt)
                improvements.append(critique[:200])
                print(f"💭 Critique: {critique[:100]}...")
                
                # Check if good enough
                if score >= 0.9:  # 90% success threshold
                    print("✅ Solution meets criteria!")
                    break
        
        return {
            "best_solution": best_solution,
            "best_score": best_score,
            "rounds": round + 1,
            "improvements": improvements
        }


def demonstrate_agent():
    """Demonstrate agent capabilities"""
    agent = GPTOSSAgent()
    
    print("\n" + "="*60)
    print("GPT-OSS AGENT DEMONSTRATION")
    print("="*60)
    
    # Demo 1: Math problem with Python tool
    print("\n📐 Demo 1: Solve Math Problem")
    result = agent.run("Find all prime numbers between 1 and 20")
    
    # Demo 2: Code analysis
    print("\n🔍 Demo 2: Analyze Code")
    result = agent.run("Analyze this code for quality: def factorial(n): return 1 if n==0 else n*factorial(n-1)")
    
    # Demo 3: Self-improving solution
    print("\n🔄 Demo 3: Self-Improving")
    
    def code_quality_criteria(solution: str) -> float:
        """Score code quality (0-1)"""
        score = 0.0
        if "def" in solution: score += 0.2
        if "#" in solution or '"""' in solution: score += 0.2  # Comments
        if "return" in solution: score += 0.2
        if len(solution) > 50: score += 0.2
        if "test" in solution.lower(): score += 0.2
        return score
    
    result = agent.self_improve(
        "Write a Python function to check if a number is prime",
        code_quality_criteria
    )
    
    print(f"\n🏆 Best solution (score: {result['best_score']}):")
    print(result["best_solution"])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-OSS Agent Framework")
    parser.add_argument("--model", default="gpt-oss:latest", help="Model name")
    parser.add_argument("--task", help="Task for agent to complete")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--self-improve", action="store_true", help="Enable self-improvement")
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_agent()
    elif args.task:
        agent = GPTOSSAgent(args.model)
        
        if args.self_improve:
            # Simple quality criteria
            def criteria(solution):
                return min(1.0, len(solution) / 100)  # Longer = better (simplified)
            
            result = agent.self_improve(args.task, criteria)
            print(f"\nFinal result: {result}")
        else:
            result = agent.run(args.task)
            print(f"\nFinal result: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()