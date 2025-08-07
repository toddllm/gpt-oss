#!/bin/bash

# Quick Ollama Test Script
# Fast validation that model is working correctly

set -e

MODEL="${1:-gpt-oss:latest}"

echo "🧪 Quick Test for $MODEL"
echo "========================"
echo ""

# Test 1: Basic response
echo "Test 1: Basic greeting"
echo -n "Response: "
echo "Hello" | ollama run "$MODEL" 2>/dev/null | head -n 2
echo ""

# Test 2: Math
echo "Test 2: Simple math (2+2)"
echo -n "Response: "
echo "What is 2+2? Answer with just the number." | ollama run "$MODEL" 2>/dev/null | head -n 2
echo ""

# Test 3: Completion
echo "Test 3: Text completion"
echo -n "Response: "
echo "Complete this: The capital of France is" | ollama run "$MODEL" 2>/dev/null | head -n 2
echo ""

echo "✅ Quick test complete!"