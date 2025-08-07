#!/bin/bash

# Ollama Benchmarking Script
# Measures performance metrics for Ollama models

set -e

# Default values
MODEL="${1:-gpt-oss:latest}"
NUM_RUNS=10
OUTPUT_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}       OLLAMA BENCHMARK SUITE${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Model: $MODEL"
echo "Number of runs: $NUM_RUNS"
echo "Output file: $OUTPUT_FILE"
echo ""

# Function to measure response time
measure_response_time() {
    local prompt="$1"
    local start=$(date +%s.%N)
    
    echo "$prompt" | ollama run "$MODEL" > /dev/null 2>&1
    
    local end=$(date +%s.%N)
    echo "$end - $start" | bc
}

# Function to measure tokens per second
measure_throughput() {
    local prompt="$1"
    local temp_file=$(mktemp)
    
    # Run with verbose flag to get token statistics
    echo "$prompt" | ollama run "$MODEL" --verbose > "$temp_file" 2>&1
    
    # Try to extract token information from output
    if grep -q "tokens" "$temp_file"; then
        tokens=$(grep -oP '\d+(?= tokens)' "$temp_file" | tail -1)
        time=$(grep -oP '\d+\.\d+(?=s)' "$temp_file" | tail -1)
        if [[ -n "$tokens" && -n "$time" ]]; then
            echo "scale=2; $tokens / $time" | bc
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
    
    rm -f "$temp_file"
}

# Start benchmarking
echo -e "${YELLOW}Starting benchmark tests...${NC}" | tee "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Test 1: Simple response time
echo "Test 1: Simple Query Response Time" | tee -a "$OUTPUT_FILE"
echo "Prompt: 'Hello, how are you?'" | tee -a "$OUTPUT_FILE"
total_time=0
for i in $(seq 1 $NUM_RUNS); do
    echo -n "  Run $i/$NUM_RUNS: "
    time=$(measure_response_time "Hello, how are you?")
    echo "${time}s"
    total_time=$(echo "$total_time + $time" | bc)
done
avg_time=$(echo "scale=3; $total_time / $NUM_RUNS" | bc)
echo "Average response time: ${avg_time}s" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Test 2: Math problem
echo "Test 2: Math Problem Solving" | tee -a "$OUTPUT_FILE"
echo "Prompt: 'What is 1234 * 5678?'" | tee -a "$OUTPUT_FILE"
total_time=0
for i in $(seq 1 $NUM_RUNS); do
    echo -n "  Run $i/$NUM_RUNS: "
    time=$(measure_response_time "What is 1234 * 5678?")
    echo "${time}s"
    total_time=$(echo "$total_time + $time" | bc)
done
avg_time=$(echo "scale=3; $total_time / $NUM_RUNS" | bc)
echo "Average response time: ${avg_time}s" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Test 3: Code generation
echo "Test 3: Code Generation" | tee -a "$OUTPUT_FILE"
echo "Prompt: 'Write a Python function to sort a list'" | tee -a "$OUTPUT_FILE"
total_time=0
for i in $(seq 1 3); do  # Fewer runs for longer prompts
    echo -n "  Run $i/3: "
    time=$(measure_response_time "Write a Python function to sort a list")
    echo "${time}s"
    total_time=$(echo "$total_time + $time" | bc)
done
avg_time=$(echo "scale=3; $total_time / 3" | bc)
echo "Average response time: ${avg_time}s" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Test 4: Context handling (longer prompt)
echo "Test 4: Long Context Handling" | tee -a "$OUTPUT_FILE"
long_prompt="Summarize this: The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet. It has been used for decades to test typewriters and computer keyboards. The phrase is short but comprehensive."
echo "Prompt: Long text summarization" | tee -a "$OUTPUT_FILE"
total_time=0
for i in $(seq 1 3); do
    echo -n "  Run $i/3: "
    time=$(measure_response_time "$long_prompt")
    echo "${time}s"
    total_time=$(echo "$total_time + $time" | bc)
done
avg_time=$(echo "scale=3; $total_time / 3" | bc)
echo "Average response time: ${avg_time}s" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Test 5: Throughput test
echo "Test 5: Throughput Measurement" | tee -a "$OUTPUT_FILE"
echo "Measuring tokens per second..." | tee -a "$OUTPUT_FILE"
throughput=$(measure_throughput "Explain quantum computing in 100 words")
echo "Throughput: $throughput tokens/second" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# System information
echo "=========================================" | tee -a "$OUTPUT_FILE"
echo "System Information:" | tee -a "$OUTPUT_FILE"
echo "=========================================" | tee -a "$OUTPUT_FILE"
echo "Date: $(date)" | tee -a "$OUTPUT_FILE"
echo "Ollama Version: $(ollama --version)" | tee -a "$OUTPUT_FILE"
echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)" | tee -a "$OUTPUT_FILE"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')" | tee -a "$OUTPUT_FILE"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "$OUTPUT_FILE"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)" | tee -a "$OUTPUT_FILE"
else
    echo "GPU: Not available" | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"
echo -e "${GREEN}✓ Benchmark complete! Results saved to $OUTPUT_FILE${NC}"