# Technical Report: GPT-OSS-20B Implementation Strategies

## Current Status
- **Download Progress**: 37GB of ~40GB (47% of files, 9/19 complete)
- **Process Status**: Active (PID 9015)
- **Remaining Files**: 2 incomplete downloads
- **Environment**: PyTorch 2.6.0, Triton 3.2.0, CUDA 12.4, RTX 3090 (24GB VRAM)

## Implementation Strategies After Download Completes

### Strategy 1: Transformers with Adaptive Loading

This approach tries multiple quantization strategies to find the optimal configuration for available hardware.

```python
#!/usr/bin/env python3
"""
Adaptive GPT-OSS-20B loader that automatically selects best configuration
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

# Disable MXFP4 warnings (not supported on RTX 3090)
os.environ['TRANSFORMERS_NO_MXFP4'] = '1'

class GPTOSSLoader:
    def __init__(self, model_id="openai/gpt-oss-20b"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.quantization_type = None
        
    def check_environment(self):
        """Verify environment and hardware"""
        info = {
            'pytorch': torch.__version__,
            'cuda': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        }
        
        # Check for required libraries
        try:
            import bitsandbytes
            info['bitsandbytes'] = True
        except ImportError:
            info['bitsandbytes'] = False
            
        return info
    
    def load_tokenizer(self):
        """Load and configure tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            local_files_only=True,  # Use cached files
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def try_load_4bit(self):
        """Attempt 4-bit quantization loading"""
        try:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.quantization_type = "4-bit NF4"
            return True
        except Exception as e:
            print(f"4-bit loading failed: {str(e)[:100]}")
            self.cleanup()
            return False
    
    def try_load_8bit(self):
        """Attempt 8-bit quantization loading"""
        try:
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.quantization_type = "8-bit"
            return True
        except Exception as e:
            print(f"8-bit loading failed: {str(e)[:100]}")
            self.cleanup()
            return False
    
    def try_load_bfloat16(self):
        """Attempt bfloat16 loading (no quantization)"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.quantization_type = "bfloat16"
            return True
        except Exception as e:
            print(f"bfloat16 loading failed: {str(e)[:100]}")
            self.cleanup()
            return False
    
    def try_load_float16(self):
        """Attempt float16 loading as fallback"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.quantization_type = "float16"
            return True
        except Exception as e:
            print(f"float16 loading failed: {str(e)[:100]}")
            self.cleanup()
            return False
    
    def cleanup(self):
        """Clean up memory after failed load attempt"""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_model(self):
        """Load model with automatic strategy selection"""
        env = self.check_environment()
        
        print(f"Environment: PyTorch {env['pytorch']}, GPU: {env['gpu']}, VRAM: {env['vram_gb']:.2f}GB")
        
        # Load tokenizer first
        print("Loading tokenizer...")
        self.load_tokenizer()
        
        # Try loading strategies in order of preference
        strategies = []
        
        if env['bitsandbytes']:
            strategies.append(("4-bit quantization", self.try_load_4bit))
            strategies.append(("8-bit quantization", self.try_load_8bit))
        
        strategies.append(("bfloat16", self.try_load_bfloat16))
        strategies.append(("float16", self.try_load_float16))
        
        for name, method in strategies:
            print(f"\nTrying {name}...")
            if method():
                print(f"✓ Successfully loaded with {self.quantization_type}")
                self.print_memory_usage()
                return True
        
        raise RuntimeError("Failed to load model with any strategy")
    
    def print_memory_usage(self):
        """Print current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\nGPU Memory:")
            print(f"  Allocated: {allocated:.2f}GB")
            print(f"  Reserved: {reserved:.2f}GB")
            print(f"  Free: {total - reserved:.2f}GB")
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

# Usage
if __name__ == "__main__":
    loader = GPTOSSLoader()
    
    try:
        # Load model
        loader.load_model()
        
        # Test generation
        test_prompts = [
            "The future of artificial intelligence is",
            "In quantum computing, the main challenge is",
            "Python is a programming language that",
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = loader.generate(prompt, max_new_tokens=50)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
```

### Strategy 2: vLLM Implementation (Optimized for Production)

vLLM provides better memory management and throughput for production deployments.

```python
#!/usr/bin/env python3
"""
vLLM implementation for GPT-OSS-20B with automatic optimization
"""

import os
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

class VLLMGPTOSSServer:
    def __init__(self, model_id="openai/gpt-oss-20b"):
        self.model_id = model_id
        self.llm = None
        self.sampling_params = None
        
    def calculate_optimal_config(self):
        """Calculate optimal vLLM configuration based on hardware"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for vLLM")
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Configuration based on GPU memory
        config = {
            'model': self.model_id,
            'trust_remote_code': True,
            'dtype': 'auto',  # Let vLLM choose
            'download_dir': os.path.expanduser("~/.cache/huggingface/hub"),
        }
        
        if gpu_memory_gb < 24:
            # For GPUs with less than 24GB
            config.update({
                'max_model_len': 1024,
                'gpu_memory_utilization': 0.95,
                'quantization': 'awq',  # Use AWQ quantization if available
            })
        elif gpu_memory_gb < 48:
            # For RTX 3090, A5000, etc (24GB)
            config.update({
                'max_model_len': 2048,
                'gpu_memory_utilization': 0.90,
            })
        else:
            # For A100, H100, etc
            config.update({
                'max_model_len': 4096,
                'gpu_memory_utilization': 0.85,
            })
        
        # Additional optimizations
        config.update({
            'disable_custom_all_reduce': True,  # For single GPU
            'enforce_eager': False,  # Use CUDA graphs for better performance
            'enable_prefix_caching': True,  # Cache common prefixes
        })
        
        return config
    
    def initialize(self):
        """Initialize vLLM with optimal settings"""
        config = self.calculate_optimal_config()
        
        print("Initializing vLLM with configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        try:
            self.llm = LLM(**config)
            
            # Default sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.9,
                max_tokens=100,
            )
            
            print("✓ vLLM initialized successfully")
            
        except Exception as e:
            # Fallback configuration for OOM errors
            if "out of memory" in str(e).lower():
                print("Retrying with reduced memory configuration...")
                config['max_model_len'] = 512
                config['gpu_memory_utilization'] = 0.95
                self.llm = LLM(**config)
            else:
                raise
    
    def generate(self, prompts, sampling_params=None):
        """Generate responses for multiple prompts"""
        if sampling_params is None:
            sampling_params = self.sampling_params
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                'prompt': output.prompt,
                'generated_text': output.outputs[0].text,
                'finish_reason': output.outputs[0].finish_reason,
            })
        
        return results
    
    def generate_stream(self, prompt, sampling_params=None):
        """Generate with streaming output"""
        if sampling_params is None:
            sampling_params = self.sampling_params
        
        # vLLM doesn't support true streaming in the same way, 
        # but we can simulate it with small chunks
        sampling_params.max_tokens = 10  # Generate in small chunks
        
        full_response = ""
        for _ in range(10):  # Generate up to 100 tokens total
            outputs = self.llm.generate([prompt + full_response], sampling_params)
            new_text = outputs[0].outputs[0].text
            if not new_text or outputs[0].outputs[0].finish_reason == "stop":
                break
            full_response += new_text
            yield new_text
    
    def benchmark(self):
        """Run performance benchmark"""
        import time
        
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the key differences between machine learning and deep learning?",
            "Describe the process of photosynthesis.",
            "How does blockchain technology work?",
        ]
        
        # Warmup
        self.generate(["Warmup prompt"], SamplingParams(max_tokens=10))
        
        # Benchmark
        start = time.time()
        results = self.generate(
            test_prompts,
            SamplingParams(temperature=0.8, top_p=0.9, max_tokens=50)
        )
        end = time.time()
        
        total_tokens = sum(len(r['generated_text'].split()) for r in results)
        throughput = total_tokens / (end - start)
        
        print(f"\nBenchmark Results:")
        print(f"  Prompts: {len(test_prompts)}")
        print(f"  Total time: {end - start:.2f}s")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        
        return results
    
    def cleanup(self):
        """Clean up vLLM resources"""
        if self.llm is not None:
            del self.llm
            destroy_model_parallel()
            torch.cuda.empty_cache()

# Usage
if __name__ == "__main__":
    server = VLLMGPTOSSServer()
    
    try:
        # Initialize
        server.initialize()
        
        # Single generation
        results = server.generate(["What is the meaning of life?"])
        print(f"\nGenerated: {results[0]['generated_text']}")
        
        # Batch generation
        batch_prompts = [
            "The future of AI is",
            "Python is best for",
            "Quantum computers will",
        ]
        
        batch_results = server.generate(batch_prompts)
        for result in batch_results:
            print(f"\nPrompt: {result['prompt']}")
            print(f"Response: {result['generated_text']}")
        
        # Run benchmark
        server.benchmark()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.cleanup()
```

### Strategy 3: FastAPI Server with Auto-Recovery

Production-ready API server with automatic error recovery and monitoring.

```python
#!/usr/bin/env python3
"""
Production FastAPI server for GPT-OSS-20B with health checks and auto-recovery
"""

import os
import gc
import asyncio
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    stream: bool = False

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    model: str
    quantization: str
    tokens_generated: int
    generation_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    quantization_type: Optional[str]
    uptime_seconds: float
    total_requests: int
    errors_count: int

# Model Manager
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.quantization_type = None
        self.model_id = "openai/gpt-oss-20b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = datetime.now()
        self.total_requests = 0
        self.error_count = 0
        self.loading = False
        
    async def initialize(self):
        """Async model initialization"""
        if self.loading:
            return
        
        self.loading = True
        try:
            print(f"[{datetime.now()}] Initializing model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=True,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try loading with different strategies
            loaded = False
            
            # Try 4-bit first (smallest memory footprint)
            if not loaded:
                try:
                    print("Attempting 4-bit quantization...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        load_in_4bit=True,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    self.quantization_type = "4-bit"
                    loaded = True
                except Exception as e:
                    print(f"4-bit failed: {e}")
                    self.cleanup()
            
            # Fallback to bfloat16
            if not loaded:
                try:
                    print("Attempting bfloat16...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    self.quantization_type = "bfloat16"
                    loaded = True
                except Exception as e:
                    print(f"bfloat16 failed: {e}")
                    raise
            
            print(f"[{datetime.now()}] Model loaded with {self.quantization_type}")
            
        finally:
            self.loading = False
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        """Generate text from request"""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        self.total_requests += 1
        start_time = datetime.now()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    top_p=request.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_only = generated_text[len(request.prompt):].strip()
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "generated_text": generated_only,
                "prompt": request.prompt,
                "model": self.model_id,
                "quantization": self.quantization_type,
                "tokens_generated": len(self.tokenizer.encode(generated_only)),
                "generation_time": generation_time,
            }
            
        except Exception as e:
            self.error_count += 1
            print(f"Generation error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "status": "healthy" if self.model else "loading" if self.loading else "not_loaded",
            "model_loaded": self.model is not None,
            "gpu_memory_used_gb": gpu_memory_used,
            "gpu_memory_total_gb": gpu_memory_total,
            "quantization_type": self.quantization_type,
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "errors_count": self.error_count,
        }

# Create model manager instance
model_manager = ModelManager()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting GPT-OSS-20B API Server...")
    await model_manager.initialize()
    yield
    # Shutdown
    print("Shutting down...")
    model_manager.cleanup()

app = FastAPI(
    title="GPT-OSS-20B API",
    description="Production API for GPT-OSS-20B model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return model_manager.get_health()

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt"""
    result = await model_manager.generate(request)
    return result

@app.post("/generate_stream")
async def generate_stream(request: GenerationRequest):
    """Stream generated text (SSE)"""
    # Implementation for streaming responses
    # This would require SSE (Server-Sent Events) setup
    raise HTTPException(status_code=501, detail="Streaming not yet implemented")

@app.post("/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (useful after OOM errors)"""
    background_tasks.add_task(model_manager.initialize)
    return {"message": "Model reload initiated"}

# Run server
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
```

### Strategy 4: Monitoring and Testing Suite

Comprehensive testing and monitoring utilities.

```python
#!/usr/bin/env python3
"""
Testing and monitoring suite for GPT-OSS-20B deployment
"""

import os
import time
import json
import psutil
import requests
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

class ModelTester:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        
    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def test_generation(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Test generation endpoint"""
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 50),
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.9),
        }
        
        try:
            start = time.time()
            response = requests.post(f"{self.api_url}/generate", json=payload)
            end = time.time()
            
            result = response.json()
            result["request_time"] = end - start
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_throughput(self, num_requests: int = 10) -> Dict[str, Any]:
        """Benchmark API throughput"""
        test_prompts = [
            "Explain the concept of",
            "Write a function to",
            "What are the benefits of",
            "How does one implement",
            "Describe the process of",
        ]
        
        results = []
        start_time = time.time()
        
        for i in range(num_requests):
            prompt = test_prompts[i % len(test_prompts)] + f" test {i}"
            result = self.test_generation(prompt, max_tokens=30)
            results.append(result)
            
            if "error" not in result:
                print(f"Request {i+1}/{num_requests}: {result.get('generation_time', 'N/A')}s")
        
        total_time = time.time() - start_time
        successful = [r for r in results if "error" not in r]
        
        stats = {
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": num_requests - len(successful),
            "total_time": total_time,
            "avg_request_time": total_time / num_requests,
            "requests_per_second": num_requests / total_time,
        }
        
        if successful:
            gen_times = [r["generation_time"] for r in successful]
            stats.update({
                "avg_generation_time": np.mean(gen_times),
                "min_generation_time": np.min(gen_times),
                "max_generation_time": np.max(gen_times),
                "p95_generation_time": np.percentile(gen_times, 95),
            })
        
        return stats
    
    def stress_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run stress test for specified duration"""
        print(f"Running stress test for {duration_seconds} seconds...")
        
        start_time = time.time()
        results = []
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            prompt = f"Generate text about topic {request_count}"
            result = self.test_generation(prompt, max_tokens=50)
            results.append({
                "timestamp": time.time() - start_time,
                "success": "error" not in result,
                "generation_time": result.get("generation_time", None),
            })
            request_count += 1
            
            # Brief pause to avoid overwhelming
            time.sleep(0.1)
        
        successful = [r for r in results if r["success"]]
        
        return {
            "duration": duration_seconds,
            "total_requests": request_count,
            "successful_requests": len(successful),
            "success_rate": len(successful) / request_count if request_count > 0 else 0,
            "avg_generation_time": np.mean([r["generation_time"] for r in successful if r["generation_time"]]),
        }
    
    def test_edge_cases(self) -> List[Dict[str, Any]]:
        """Test edge cases and error handling"""
        test_cases = [
            {
                "name": "Empty prompt",
                "prompt": "",
                "expected": "error",
            },
            {
                "name": "Very long prompt",
                "prompt": "test " * 1000,
                "expected": "truncation",
            },
            {
                "name": "Special characters",
                "prompt": "Test with émojis 🤖 and symbols ©®™",
                "expected": "success",
            },
            {
                "name": "Code generation",
                "prompt": "def fibonacci(n):",
                "expected": "success",
            },
            {
                "name": "Max tokens limit",
                "prompt": "Generate a story",
                "max_tokens": 2048,
                "expected": "success",
            },
        ]
        
        results = []
        for test in test_cases:
            print(f"Testing: {test['name']}")
            result = self.test_generation(
                test["prompt"],
                max_tokens=test.get("max_tokens", 50)
            )
            
            results.append({
                "test": test["name"],
                "passed": ("error" in result) == (test["expected"] == "error"),
                "result": result,
            })
        
        return results

class SystemMonitor:
    def __init__(self):
        self.metrics = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        import GPUtil
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / 1024**3,
        }
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_percent": gpu.memoryUtil * 100,
                    "gpu_memory_used_gb": gpu.memoryUsed / 1024,
                    "gpu_temperature": gpu.temperature,
                })
        except:
            pass
        
        self.metrics.append(metrics)
        return metrics
    
    def plot_metrics(self, duration_seconds: int = 60):
        """Plot system metrics over time"""
        print(f"Monitoring system for {duration_seconds} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            self.collect_metrics()
            time.sleep(1)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        timestamps = range(len(self.metrics))
        
        # CPU usage
        axes[0, 0].plot(timestamps, [m["cpu_percent"] for m in self.metrics])
        axes[0, 0].set_title("CPU Usage (%)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Usage (%)")
        
        # Memory usage
        axes[0, 1].plot(timestamps, [m["memory_percent"] for m in self.metrics])
        axes[0, 1].set_title("RAM Usage (%)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Usage (%)")
        
        # GPU utilization
        if "gpu_utilization" in self.metrics[0]:
            axes[1, 0].plot(timestamps, [m["gpu_utilization"] for m in self.metrics])
            axes[1, 0].set_title("GPU Utilization (%)")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Usage (%)")
            
            # GPU memory
            axes[1, 1].plot(timestamps, [m["gpu_memory_used_gb"] for m in self.metrics])
            axes[1, 1].set_title("GPU Memory (GB)")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Memory (GB)")
        
        plt.tight_layout()
        plt.savefig("system_metrics.png")
        print("Metrics saved to system_metrics.png")
        
        return self.metrics

# Main testing routine
if __name__ == "__main__":
    print("GPT-OSS-20B Testing Suite")
    print("=" * 50)
    
    tester = ModelTester()
    monitor = SystemMonitor()
    
    # 1. Health check
    print("\n1. Health Check")
    health = tester.test_health()
    print(json.dumps(health, indent=2))
    
    # 2. Basic generation test
    print("\n2. Basic Generation Test")
    result = tester.test_generation("The future of AI is")
    if "error" not in result:
        print(f"Generated: {result['generated_text'][:100]}...")
        print(f"Time: {result['generation_time']:.2f}s")
    else:
        print(f"Error: {result['error']}")
    
    # 3. Throughput benchmark
    print("\n3. Throughput Benchmark")
    benchmark = tester.benchmark_throughput(10)
    print(json.dumps(benchmark, indent=2))
    
    # 4. Edge case testing
    print("\n4. Edge Case Testing")
    edge_results = tester.test_edge_cases()
    for result in edge_results:
        status = "✓" if result["passed"] else "✗"
        print(f"{status} {result['test']}")
    
    # 5. System monitoring (optional)
    if input("\nRun system monitoring? (y/n): ").lower() == 'y':
        monitor.plot_metrics(30)
```

## Deployment Recommendations

### For Development/Testing
- Use Strategy 1 (Adaptive Loading) for flexibility
- Run with bfloat16 for best quality/performance balance
- Monitor GPU memory usage closely

### For Production
- Deploy Strategy 3 (FastAPI Server) with Strategy 2 (vLLM) backend
- Use 4-bit quantization for maximum throughput
- Implement health checks and auto-recovery
- Set up monitoring with Strategy 4

### Hardware Considerations
- **RTX 3090 (24GB)**: Can run full model in bfloat16 or 4-bit quantized
- **Memory Requirements**:
  - 4-bit: ~10-12GB VRAM
  - 8-bit: ~20GB VRAM  
  - bfloat16: ~40GB VRAM (requires model parallelism)
  - float16: ~40GB VRAM

### Performance Expectations
- **4-bit quantization**: 5-10 tokens/sec, slight quality degradation
- **8-bit quantization**: 3-5 tokens/sec, minimal quality loss
- **bfloat16**: 2-3 tokens/sec, full quality
- **vLLM**: 2-3x throughput improvement over transformers

## Next Steps

1. Wait for download completion (~40GB total, currently at 37GB)
2. Run health check: `python run_after_download.py`
3. Choose deployment strategy based on use case
4. Implement monitoring and testing
5. Optimize based on performance metrics