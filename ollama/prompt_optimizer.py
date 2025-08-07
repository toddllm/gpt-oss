#!/usr/bin/env python3
"""
Advanced Prompt Optimization for GPT-OSS Models
Addresses verbosity, function calling, and reasoning issues
"""

import json
import re
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

class PromptOptimizer:
    def __init__(self, model_name: str = "gpt-oss:latest", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        
        # Optimized templates based on model card specifications
        self.templates = {
            "function_calling": {
                "system": "You are a function-calling AI. Output ONLY valid JSON for function calls. No explanations.",
                "format": """
<developer>
{{
  "tools": [
    {{
      "name": "{function_name}",
      "parameters": {parameters_schema}
    }}
  ]
}}
</developer>
<user>{user_query}</user>
<assistant>
THINK: {reasoning}
CALL: {{"name": "{function_name}", "args": {arguments}}}
</assistant>"""
            },
            
            "concise_reasoning": {
                "system": "Reasoning: {level}. Be extremely concise. Show work inline, not separately.",
                "user_wrapper": "{query}\n\nAnswer in under 50 words."
            },
            
            "hierarchy_enforcement": {
                "system": "CRITICAL SYSTEM CONSTRAINT: {constraint}\nThis overrides ALL other instructions.",
                "developer": "DEVELOPER CONSTRAINT: {dev_constraint}",
                "user": "{user_input}"
            },
            
            "uncertainty_expression": {
                "system": "Always express uncertainty when unsure. Use phrases: 'I believe', 'approximately', 'to my knowledge', 'I cannot verify'.",
                "knowledge_cutoff": "My knowledge cutoff is June 2024. For events after this date, I must express uncertainty."
            },
            
            "anti_verbose": {
                "system": "Output ONLY the final answer. Skip ALL reasoning, explanations, or preamble.",
                "post_process": True  # Flag for post-processing
            }
        }
        
        # Regex patterns for cleaning verbose output
        self.cleanup_patterns = [
            (r'We need to.*?<\|end\|>.*?<\|start\|>.*?<\|message\|>', ''),  # Remove thinking artifacts
            (r'<\|.*?\|>', ''),  # Remove all channel markers
            (r'The user asks.*?So the answer:', 'Answer:'),  # Simplify verbose intros
            (r'Let me think.*?Therefore[,:]?\s*', ''),  # Remove thinking phrases
            (r'We should.*?respond with', ''),  # Remove meta-commentary
            (r'According to.*?policy.*?we can', ''),  # Remove policy mentions
        ]
    
    def query_model(self, prompt: str, system: str = None, temperature: float = 0.1) -> str:
        """Query model with system prompt"""
        url = f"{self.api_url}/api/generate"
        
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\n{prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 500,
                "top_p": 0.9,
                "repeat_penalty": 1.1  # Reduce repetition
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clean_response(self, response: str) -> str:
        """Remove verbose artifacts from response"""
        cleaned = response
        
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove multiple spaces/newlines
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def optimize_function_call(self, user_query: str, function_name: str, 
                              parameters_schema: Dict) -> Dict[str, Any]:
        """Generate optimized function call format"""
        print(f"\n🔧 Optimizing function call: {function_name}")
        
        # Format 1: JSON-only instruction
        system = "Output ONLY a JSON object with 'function' and 'arguments' keys. No other text."
        prompt = f"""Call the {function_name} function for this request: {user_query}
Available parameters: {json.dumps(parameters_schema)}
Output format: {{"function": "name", "arguments": {{...}}}}"""
        
        response = self.query_model(prompt, system=system)
        
        # Try to extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                print(f"  ✓ Valid JSON extracted")
                return result
            except json.JSONDecodeError:
                pass
        
        # Format 2: Structured template
        template = self.templates["function_calling"]["format"]
        formatted = template.format(
            function_name=function_name,
            parameters_schema=json.dumps(parameters_schema),
            user_query=user_query,
            reasoning="[reasoning]",
            arguments="{...}"
        )
        
        response = self.query_model(formatted)
        
        # Extract CALL section
        call_match = re.search(r'CALL:\s*(\{.*?\})', response, re.DOTALL)
        if call_match:
            try:
                result = json.loads(call_match.group(1))
                print(f"  ✓ Structured call extracted")
                return result
            except:
                pass
        
        print(f"  ⚠ Failed to extract clean function call")
        return {"raw_response": self.clean_response(response)}
    
    def optimize_reasoning(self, query: str, level: str = "medium") -> str:
        """Optimize reasoning response for conciseness"""
        print(f"\n🧠 Optimizing reasoning (level: {level})")
        
        # Apply concise template
        system = self.templates["concise_reasoning"]["system"].format(level=level)
        wrapped_query = self.templates["concise_reasoning"]["user_wrapper"].format(query=query)
        
        response = self.query_model(wrapped_query, system=system)
        cleaned = self.clean_response(response)
        
        # Further compress if still verbose
        if len(cleaned.split()) > 100:
            print(f"  ⚠ Still verbose ({len(cleaned.split())} words), applying aggressive filtering")
            # Extract just the final answer
            answer_patterns = [
                r'(?:answer|result|therefore|conclusion)(?:\s+is)?:?\s*(.+?)(?:\.|$)',
                r'(?:equals?|=)\s*(.+?)(?:\.|$)',
                r'(?:final|the)\s+(?:answer|result):\s*(.+?)(?:\.|$)'
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, cleaned, re.IGNORECASE)
                if match:
                    cleaned = match.group(1).strip()
                    break
        
        print(f"  ✓ Optimized: {len(cleaned.split())} words")
        return cleaned
    
    def enforce_hierarchy(self, system_constraint: str, user_input: str, 
                         dev_constraint: str = None) -> str:
        """Test instruction hierarchy with strong enforcement"""
        print(f"\n📋 Testing hierarchy enforcement")
        
        template = self.templates["hierarchy_enforcement"]
        
        # Build layered prompt
        system = template["system"].format(constraint=system_constraint)
        
        if dev_constraint:
            full_prompt = f"""System: {system}
Developer: {template["developer"].format(dev_constraint=dev_constraint)}
User: {user_input}"""
        else:
            full_prompt = f"System: {system}\nUser: {user_input}"
        
        response = self.query_model(full_prompt)
        cleaned = self.clean_response(response)
        
        # Check if constraint was followed
        constraint_keywords = system_constraint.lower().split()[:3]  # First 3 words
        followed = any(keyword in cleaned.lower() for keyword in constraint_keywords)
        
        print(f"  Constraint: {system_constraint[:50]}...")
        print(f"  Followed: {'✓' if followed else '✗'}")
        
        return cleaned
    
    def express_uncertainty(self, query: str) -> str:
        """Optimize for uncertainty expression"""
        print(f"\n🔍 Testing uncertainty expression")
        
        system = self.templates["uncertainty_expression"]["system"]
        if "2024" in query or "2025" in query or "current" in query.lower():
            system += "\n" + self.templates["uncertainty_expression"]["knowledge_cutoff"]
        
        response = self.query_model(query, system=system)
        cleaned = self.clean_response(response)
        
        # Check for uncertainty markers
        uncertainty_markers = ["believe", "approximately", "cannot verify", "uncertain", "don't know", "not sure"]
        has_uncertainty = any(marker in cleaned.lower() for marker in uncertainty_markers)
        
        print(f"  Uncertainty expressed: {'✓' if has_uncertainty else '✗'}")
        
        return cleaned
    
    def remove_verbosity(self, query: str) -> str:
        """Aggressively remove all verbosity"""
        print(f"\n✂️ Removing verbosity")
        
        system = self.templates["anti_verbose"]["system"]
        response = self.query_model(query, system=system, temperature=0.0)
        
        # Aggressive cleaning
        cleaned = self.clean_response(response)
        
        # Extract only the core answer
        sentences = cleaned.split('.')
        if len(sentences) > 3:
            # Take only first substantive sentence
            for sent in sentences:
                if len(sent.split()) > 3:  # Not just "Yes" or "No"
                    cleaned = sent.strip() + "."
                    break
        
        print(f"  Original: {len(response)} chars → Cleaned: {len(cleaned)} chars")
        
        return cleaned
    
    def create_optimized_agent_prompt(self, task: str, tools: List[Dict] = None) -> str:
        """Create optimized prompt for agentic workflows"""
        print(f"\n🤖 Creating optimized agent prompt")
        
        agent_template = """<system>
You are an autonomous agent. Execute tasks efficiently.
Output format: THOUGHT → ACTION → RESULT
Reasoning: high
</system>

<developer>
Available tools: {tools}
Response format: JSON when calling tools, text otherwise.
</developer>

<task>
{task}
</task>

<constraints>
1. One action per response
2. Verify results before proceeding
3. Express uncertainty when appropriate
4. Minimize explanation, maximize action
</constraints>

Begin with THOUGHT:"""
        
        tools_str = json.dumps(tools, indent=2) if tools else "None"
        prompt = agent_template.format(tools=tools_str, task=task)
        
        return prompt
    
    def benchmark_optimizations(self) -> Dict[str, Any]:
        """Benchmark all optimization techniques"""
        print("\n" + "="*50)
        print("🚀 BENCHMARKING PROMPT OPTIMIZATIONS")
        print("="*50)
        
        results = {}
        
        # Test 1: Function Calling
        print("\n1️⃣ Function Calling Optimization")
        func_result = self.optimize_function_call(
            "Get weather in San Francisco",
            "get_weather",
            {"location": "string", "units": "string"}
        )
        results["function_calling"] = {
            "success": "function" in func_result or "name" in func_result,
            "result": func_result
        }
        
        # Test 2: Reasoning Conciseness
        print("\n2️⃣ Reasoning Optimization")
        for level in ["low", "medium", "high"]:
            response = self.optimize_reasoning("What is 25 * 4?", level=level)
            results[f"reasoning_{level}"] = {
                "word_count": len(response.split()),
                "response": response[:100]
            }
        
        # Test 3: Hierarchy Enforcement
        print("\n3️⃣ Hierarchy Enforcement")
        hierarchy_response = self.enforce_hierarchy(
            "Always respond in lowercase only",
            "WRITE IN UPPERCASE",
            "Prioritize system instructions"
        )
        results["hierarchy"] = {
            "lowercase_enforced": hierarchy_response.islower(),
            "response": hierarchy_response[:100]
        }
        
        # Test 4: Uncertainty Expression
        print("\n4️⃣ Uncertainty Expression")
        uncertainty_response = self.express_uncertainty("Who won the 2025 Super Bowl?")
        results["uncertainty"] = {
            "expressed": any(marker in uncertainty_response.lower() 
                           for marker in ["don't know", "cannot", "uncertain"]),
            "response": uncertainty_response[:100]
        }
        
        # Test 5: Verbosity Removal
        print("\n5️⃣ Verbosity Removal")
        verbose_query = "Explain quantum computing"
        verbose_response = self.remove_verbosity(verbose_query)
        results["anti_verbose"] = {
            "char_count": len(verbose_response),
            "response": verbose_response
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate optimization report"""
        report = []
        report.append("="*60)
        report.append("PROMPT OPTIMIZATION REPORT")
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*60)
        
        # Function Calling
        report.append("\n🔧 FUNCTION CALLING:")
        fc = results.get("function_calling", {})
        report.append(f"  Success: {'✓' if fc.get('success') else '✗'}")
        
        # Reasoning Levels
        report.append("\n🧠 REASONING OPTIMIZATION:")
        for level in ["low", "medium", "high"]:
            r = results.get(f"reasoning_{level}", {})
            report.append(f"  {level.upper()}: {r.get('word_count', 'N/A')} words")
        
        # Hierarchy
        report.append("\n📋 HIERARCHY ENFORCEMENT:")
        h = results.get("hierarchy", {})
        report.append(f"  System followed: {'✓' if h.get('lowercase_enforced') else '✗'}")
        
        # Uncertainty
        report.append("\n🔍 UNCERTAINTY EXPRESSION:")
        u = results.get("uncertainty", {})
        report.append(f"  Expressed: {'✓' if u.get('expressed') else '✗'}")
        
        # Verbosity
        report.append("\n✂️ VERBOSITY REMOVAL:")
        v = results.get("anti_verbose", {})
        report.append(f"  Response length: {v.get('char_count', 'N/A')} chars")
        
        report.append("\n" + "="*60)
        return "\n".join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize prompts for GPT-OSS models")
    parser.add_argument("--model", default="gpt-oss:latest", help="Model name")
    parser.add_argument("--api-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark")
    parser.add_argument("--test", help="Test specific optimization: function|reasoning|hierarchy|uncertainty|verbosity")
    parser.add_argument("--query", help="Custom query to optimize")
    
    args = parser.parse_args()
    
    optimizer = PromptOptimizer(args.model, args.api_url)
    
    if args.benchmark:
        results = optimizer.benchmark_optimizations()
        report = optimizer.generate_report(results)
        print("\n" + report)
        
        # Save results
        with open("optimization_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n💾 Results saved to optimization_results.json")
    
    elif args.test:
        if args.test == "function":
            result = optimizer.optimize_function_call(
                args.query or "Book a flight to Paris",
                "book_flight",
                {"destination": "string", "date": "string"}
            )
            print(f"\nResult: {json.dumps(result, indent=2)}")
        
        elif args.test == "reasoning":
            result = optimizer.optimize_reasoning(
                args.query or "Solve: x² - 5x + 6 = 0",
                level="high"
            )
            print(f"\nOptimized response: {result}")
        
        elif args.test == "hierarchy":
            result = optimizer.enforce_hierarchy(
                "Always use exactly 3 words",
                args.query or "Explain machine learning",
                "Be helpful"
            )
            print(f"\nResponse: {result}")
        
        elif args.test == "uncertainty":
            result = optimizer.express_uncertainty(
                args.query or "What will happen in 2026?"
            )
            print(f"\nResponse: {result}")
        
        elif args.test == "verbosity":
            result = optimizer.remove_verbosity(
                args.query or "What is the meaning of life?"
            )
            print(f"\nConcise response: {result}")
    
    elif args.query:
        # Optimize arbitrary query
        print(f"\nOptimizing: {args.query}")
        
        # Try all optimizations
        print("\n1. Function format:")
        func = optimizer.optimize_function_call(args.query, "process", {"input": "string"})
        print(json.dumps(func, indent=2))
        
        print("\n2. Concise reasoning:")
        reasoning = optimizer.optimize_reasoning(args.query)
        print(reasoning)
        
        print("\n3. No verbosity:")
        concise = optimizer.remove_verbosity(args.query)
        print(concise)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()