#!/bin/bash

# Ollama Health Check Script
# Verifies Ollama installation and model availability

set -e

echo "========================================="
echo "       OLLAMA HEALTH CHECK"
echo "========================================="
echo ""

# Check if Ollama is installed
echo "1. Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    VERSION=$(ollama --version)
    echo "   ✓ Ollama installed: $VERSION"
else
    echo "   ✗ Ollama not found in PATH"
    exit 1
fi
echo ""

# Check if Ollama service is running
echo "2. Checking Ollama service status..."
if pgrep -x "ollama" > /dev/null; then
    echo "   ✓ Ollama service is running"
else
    echo "   ⚠ Ollama service not running. Starting it..."
    ollama serve > /dev/null 2>&1 &
    sleep 2
    if pgrep -x "ollama" > /dev/null; then
        echo "   ✓ Ollama service started successfully"
    else
        echo "   ✗ Failed to start Ollama service"
        exit 1
    fi
fi
echo ""

# List available models
echo "3. Available models:"
ollama list | while IFS= read -r line; do
    echo "   $line"
done
echo ""

# Check GPU availability
echo "4. GPU Status:"
if nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
    echo "   ✓ GPU detected: $GPU_NAME"
    echo "   • Total Memory: ${GPU_MEM} MB"
    echo "   • Current Utilization: ${GPU_UTIL}%"
else
    echo "   ⚠ No NVIDIA GPU detected or nvidia-smi not available"
fi
echo ""

# Test model loading
echo "5. Testing model availability..."
MODEL_TO_TEST="gpt-oss:latest"
if ollama list | grep -q "$MODEL_TO_TEST"; then
    echo "   Testing $MODEL_TO_TEST..."
    RESPONSE=$(echo "Hi" | ollama run $MODEL_TO_TEST 2>/dev/null | head -n5)
    if [ $? -eq 0 ]; then
        echo "   ✓ Model $MODEL_TO_TEST is functional"
    else
        echo "   ✗ Model $MODEL_TO_TEST failed to respond"
    fi
else
    echo "   ⚠ Model $MODEL_TO_TEST not found"
    echo "   Available models:"
    ollama list | grep -v "NAME" | awk '{print "     • "$1}'
fi
echo ""

# Check disk space
echo "6. Disk space for models:"
MODEL_PATH="$HOME/.ollama/models"
if [ -d "$MODEL_PATH" ]; then
    DISK_USAGE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    echo "   Model storage: $DISK_USAGE"
fi
DISK_FREE=$(df -h "$HOME" | awk 'NR==2 {print $4}')
echo "   Free space: $DISK_FREE"
echo ""

echo "========================================="
echo "       HEALTH CHECK COMPLETE"
echo "========================================="