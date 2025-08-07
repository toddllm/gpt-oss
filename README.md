# GPT-OSS Testing & Optimization Suite

Advanced testing, benchmarking, and optimization tools for OpenAI's GPT-OSS models (gpt-oss-20b and gpt-oss-120b).

## 🚀 Overview

This repository contains comprehensive tools for working with GPT-OSS models, released by OpenAI on August 5, 2025. These are open-weight autoregressive Mixture-of-Experts transformers designed for reasoning, tool use, and agentic workflows.

### Model Details
- **gpt-oss-20b**: 20.9B parameters (3.6B active), optimized for consumer hardware
- **gpt-oss-120b**: 116.8B parameters (5.1B active), approaching GPT-4o-mini performance
- **License**: Apache 2.0 (fully open source)
- **Knowledge Cutoff**: June 2024

## 🦙 Ollama Integration

The `ollama/` directory contains specialized tools for running GPT-OSS models via Ollama, which provides efficient local inference with quantization support.

### Why Ollama?
- **Quantization**: Runs 20B model in 13GB (GGUF format) on consumer GPUs
- **Performance**: 80-82 tokens/second on RTX 3090
- **Ease of Use**: Simple CLI interface, no complex setup
- **Optimization**: Built-in GPU acceleration and memory management

### Ollama Tools Overview

#### Testing & Benchmarking
- **`test_gpt_oss_capabilities.py`**: Comprehensive capability testing
  - Tests reasoning levels (low/medium/high)
  - Chain-of-Thought analysis (6/6 score on logic puzzles)
  - Tool use and function calling
  - Instruction hierarchy (System > User priority)
  - Edge cases and hallucination detection
  
- **`benchmark_advanced.py`**: Academic benchmark implementations
  - AIME-style mathematics problems
  - GPQA expert-level questions
  - MMLU multitask assessment
  - Codeforces programming challenges
  - SWE-Bench software engineering tasks

#### Optimization Tools
- **`prompt_optimizer.py`**: Advanced prompt engineering
  - **87% verbosity reduction** (848 → 113 characters)
  - Clean JSON function calling format
  - Instruction hierarchy enforcement
  - Uncertainty expression training
  - Aggressive response filtering with regex

#### Agent Framework
- **`agent_framework.py`**: Autonomous agent system
  - Multi-step reasoning with tool integration
  - Self-improving loops (agent critiques own solutions)
  - Built-in tools: Python execution, web search, file operations
  - THOUGHT → ACTION → OBSERVATION → RESULT format

#### Fine-Tuning Preparation
- **`prepare_finetune.py`**: Dataset generation for model improvement
  - 7 categories targeting specific weaknesses
  - 650+ samples per run
  - Multiple formats: JSONL, Alpaca, ShareGPT
  - LoRA configuration templates
  - Ready-to-use Colab notebooks

#### Utility Scripts
- **`model_manager.sh`**: Interactive model management menu
- **`check_health.sh`**: System and model health verification
- **`benchmark.sh`**: Performance benchmarking suite
- **`quick_test.sh`**: Rapid model validation
- **`api_client.py`**: Python client for programmatic access

### Key Findings from Ollama Testing

#### Performance Metrics (RTX 3090, 24GB VRAM)
| Metric | Value |
|--------|-------|
| Token Generation | 80-82 tokens/sec |
| Response Latency | 1-3.5 seconds |
| Context Handling | 1000+ words |
| Model Size | 13GB (quantized) |
| Success Rate | 100% stability |

#### Model Characteristics
The quantized Ollama version exhibits unique behaviors:
- **"Thinking Out Loud"**: Shows internal reasoning process
- **Channel Artifacts**: `<|end|><|start|>assistant<|channel|>final<|message|>`
- **Verbose CoT**: Excellent reasoning but needs filtering
- **Strong STEM**: Good performance on technical tasks

#### Optimization Impact
| Issue | Before | After Optimization |
|-------|--------|-------------------|
| Verbosity | 500+ words | 50 words (87% reduction) |
| Function Calls | 30% valid JSON | 70% valid JSON |
| Hierarchy | 40% compliance | 80% compliance |
| Uncertainty | Rarely expressed | 60% appropriate |

## 🛠️ Installation

### Quick Start with Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull GPT-OSS model
ollama pull gpt-oss:latest

# Clone this repository
git clone https://github.com/toddllm/gpt-oss.git
cd gpt-oss

# Run health check
./ollama/check_health.sh

# Test capabilities
cd ollama
python3 test_gpt_oss_capabilities.py --model gpt-oss:latest
```

### Python Requirements
```bash
pip install requests  # For API clients
```

## 📊 Usage Examples

### 1. Reduce Response Verbosity
```bash
python3 ollama/prompt_optimizer.py --test verbosity --query "Explain quantum computing"
# Result: 87% character reduction
```

### 2. Run Autonomous Agent
```bash
python3 ollama/agent_framework.py --task "Write and optimize a prime number checker" --self-improve
# Agent will iterate and improve its solution
```

### 3. Generate Fine-Tuning Data
```bash
python3 ollama/prepare_finetune.py --samples 1000
# Creates datasets targeting model weaknesses
```

### 4. Benchmark Performance
```bash
./ollama/benchmark.sh gpt-oss:latest
# Comprehensive performance metrics
```

## 📈 Benchmark Results

### Local (Quantized) vs Official Performance
| Benchmark | Local Ollama | Official 20b | Official 120b |
|-----------|--------------|--------------|---------------|
| AIME Math | 5.5/7 reasoning | ~85% | 97% |
| GPQA Expert | 2.5/7 technical | 71.5% | 80.1% |
| MMLU | 100% (simple) | 85.3% | 90% |
| Codeforces | Good quality | 2000+ Elo | 2620 Elo |

## 🔬 Advanced Features

### Prompt Optimization Strategies
1. **Temperature 0.1**: Consistency improvement
2. **CRITICAL prefix**: Better constraint following
3. **JSON-only prompts**: Clean function calling
4. **Regex cleaning**: Remove thinking artifacts

### Agent Capabilities
- Multi-step problem solving
- Tool integration (Python, search, files)
- Self-critique and improvement
- Structured reasoning format

### Fine-Tuning Preparation
- Function calling format training
- Verbosity reduction examples
- Hierarchy enforcement data
- Uncertainty expression training
- Adversarial resistance samples

## 📁 Project Structure

```
gpt-oss/
├── README.md                   # This file
├── Modelfile                   # Ollama model configuration
├── .gitignore                  # Excludes large files
└── ollama/                     # Ollama-specific tools
    ├── test_gpt_oss_capabilities.py  # Capability testing
    ├── benchmark_advanced.py         # Academic benchmarks
    ├── prompt_optimizer.py           # Prompt optimization (87% reduction!)
    ├── agent_framework.py            # Autonomous agent
    ├── prepare_finetune.py          # Fine-tuning data generation
    ├── api_client.py                 # Ollama API client
    ├── model_manager.sh             # Model management
    ├── check_health.sh              # System health check
    ├── benchmark.sh                 # Performance benchmarking
    ├── quick_test.sh                # Rapid validation
    ├── README.md                    # Ollama tools documentation
    ├── GPT_OSS_TEST_RESULTS.md      # Comprehensive test results
    ├── ADVANCED_TESTING_SUMMARY.md  # Advanced testing overview
    └── SMOKE_TEST_RESULTS.md        # Smoke test results
```

## 🎯 Use Cases

- **Research**: Study CoT reasoning and model behavior
- **Development**: Build optimized agents and applications  
- **Education**: Learn from visible reasoning processes
- **Optimization**: Improve outputs for production use
- **Fine-Tuning**: Prepare targeted training datasets

## 🚀 Next Steps

1. **Optimize Further**: Experiment with prompt templates
2. **Build Agents**: Create domain-specific tools
3. **Fine-Tune** (Future): Use generated datasets for improvement
4. **Deploy**: Integrate optimized prompts in production

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional benchmarks
- Optimization techniques
- Agent tools and workflows
- Fine-tuning experiments
- Documentation improvements

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Resources

- [OpenAI GPT-OSS Model Card](https://openai.com/gpt-oss)
- [Ollama Documentation](https://ollama.ai/docs)
- [HuggingFace Model Weights](https://huggingface.co/openai/gpt-oss-20b)
- [GitHub Repository](https://github.com/toddllm/gpt-oss)

## ⚠️ Disclaimer

Independent project for testing and optimizing GPT-OSS models. Not affiliated with OpenAI.

---

**Status**: Active Development  
**Last Updated**: August 6, 2025  
**Main Achievement**: 87% verbosity reduction with prompt optimization