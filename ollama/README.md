# Ollama Utilities Suite

A comprehensive collection of scripts and tools for managing, testing, and benchmarking Ollama models.

## 🚀 Quick Start

After a system reboot, verify Ollama is working:
```bash
./check_health.sh
```

Quick test the gpt-oss model:
```bash
./quick_test.sh gpt-oss:latest
```

## 📁 Scripts Overview

### 1. **check_health.sh** - System Health Check
Comprehensive health check for Ollama installation.
- Verifies Ollama installation and version
- Checks service status
- Lists available models
- Reports GPU status
- Tests model responsiveness
- Shows disk usage

**Usage:**
```bash
./check_health.sh
```

### 2. **model_manager.sh** - Model Management
Interactive menu for model operations.
- List installed models
- Pull new models
- Remove models
- Show model information
- Create custom models from Modelfile
- Export/Import models
- Update all models
- Check disk usage

**Usage:**
```bash
./model_manager.sh
```

### 3. **test_models.py** - Comprehensive Testing
Python script for thorough model testing.
- Basic prompt testing (math, code, reasoning, creative)
- Performance benchmarking
- Context handling tests
- Consistency checking
- Detailed reporting

**Usage:**
```bash
# Full test suite
python3 test_models.py --model gpt-oss:latest

# Quick tests only
python3 test_models.py --model gpt-oss:latest --quick

# Save results to file
python3 test_models.py --model gpt-oss:latest --output results.json
```

### 4. **benchmark.sh** - Performance Benchmarking
Measures model performance metrics.
- Response time testing
- Throughput measurement
- Multiple test categories
- System information logging

**Usage:**
```bash
./benchmark.sh [model_name]

# Example:
./benchmark.sh gpt-oss:latest
```

### 5. **quick_test.sh** - Rapid Validation
Fast validation script for model functionality.
- Basic greeting test
- Simple math test
- Text completion test

**Usage:**
```bash
./quick_test.sh [model_name]
```

### 6. **api_client.py** - Ollama API Client
Python client for programmatic interaction with Ollama.
- Generate text
- Interactive chat mode
- List models
- Pull models
- Generate embeddings

**Usage:**
```bash
# Interactive chat
python3 api_client.py --model gpt-oss:latest chat

# Generate text
python3 api_client.py generate "Write a haiku about AI"

# List models
python3 api_client.py list

# Generate embeddings
python3 api_client.py embeddings "Hello world"
```

## 🔧 Setup

1. Make all shell scripts executable:
```bash
chmod +x *.sh
```

2. Install Python dependencies (if needed):
```bash
pip install requests
```

3. Ensure Ollama is running:
```bash
ollama serve
```

## 📊 Model Testing Workflow

### After System Reboot:
1. Check system health: `./check_health.sh`
2. Quick validation: `./quick_test.sh`
3. Run benchmarks: `./benchmark.sh`

### For New Models:
1. Pull model: Use `model_manager.sh` option 2
2. Quick test: `./quick_test.sh new-model`
3. Full test: `python3 test_models.py --model new-model`
4. Benchmark: `./benchmark.sh new-model`

### For Development:
1. Use `api_client.py` for programmatic access
2. Interactive chat: `python3 api_client.py chat`
3. Test endpoints and responses

## 🎯 Common Use Cases

### Testing gpt-oss:20b Model
```bash
# Quick validation
./quick_test.sh gpt-oss:latest

# Full testing suite
python3 test_models.py --model gpt-oss:latest

# Performance benchmark
./benchmark.sh gpt-oss:latest
```

### Managing Models
```bash
# Interactive management
./model_manager.sh

# Check all models
ollama list

# Pull new model
ollama pull llama3.1:8b
```

### API Integration
```python
from api_client import OllamaClient

client = OllamaClient()
for chunk in client.generate("gpt-oss:latest", "Hello world"):
    if "response" in chunk:
        print(chunk["response"], end="")
```

## 📝 Custom Model Creation

Create a `Modelfile`:
```dockerfile
FROM gpt-oss:latest

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM "You are a helpful coding assistant."
```

Then create the model:
```bash
ollama create my-custom-model -f Modelfile
```

## 🔍 Troubleshooting

### Ollama Service Not Running
```bash
# Start service
ollama serve

# Or run in background
nohup ollama serve > ollama.log 2>&1 &
```

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Model Not Responding
```bash
# Check model is loaded
ollama list

# Try pulling again
ollama pull gpt-oss:latest

# Check logs
journalctl -u ollama -f
```

## 📈 Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly configured
2. **Memory Management**: Monitor with `nvidia-smi` during inference
3. **Context Length**: Adjust `num_ctx` parameter for your use case
4. **Temperature**: Lower values (0.1-0.3) for consistent outputs
5. **Batch Processing**: Use API client for multiple requests

## 🛠️ Advanced Configuration

### Environment Variables
```bash
export OLLAMA_HOST=0.0.0.0:11434  # Allow remote connections
export OLLAMA_MODELS=/path/to/models  # Custom model directory
export OLLAMA_KEEP_ALIVE=5m  # Model keep-alive duration
```

### API Endpoints
- Generate: `POST http://localhost:11434/api/generate`
- Chat: `POST http://localhost:11434/api/chat`
- List: `GET http://localhost:11434/api/tags`
- Pull: `POST http://localhost:11434/api/pull`

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Model Library](https://ollama.ai/library)
- [API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

## 🤝 Contributing

Feel free to add more utilities or improve existing ones. Each script should:
- Be well-commented
- Handle errors gracefully
- Provide clear output
- Follow consistent styling

## 📄 License

These utilities are provided as-is for testing and development purposes.