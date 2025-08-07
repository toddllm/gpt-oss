#\!/usr/bin/env python3
import os
os.environ["TRANSFORMERS_NO_MXFP4"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

from vllm import LLM, SamplingParams

model_path = "/home/tdeshane/gpt-oss/gpt-oss-20b-final"

print("Loading GPT-OSS model...")
try:
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True
    )
    
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    print("Generating output...")
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        
except Exception as e:
    print(f"Error: {e}")
