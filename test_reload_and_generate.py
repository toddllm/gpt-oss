#!/usr/bin/env python3
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

LOCAL_MODEL_DIR = os.environ.get("MODEL_DIR", "/home/tdeshane/gpt-oss/gpt-oss-20b-final")
ADAPTER_DIR = os.environ.get("ADAPTER_OUTPUT_DIR", "outputs/lora_adapters")

# Load base in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    device_map={"": 0},
    offload_state_dict=False,
)

# Load adapters if present
if os.path.isdir(ADAPTER_DIR):
    print(f"Loading adapters from {ADAPTER_DIR}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

# Generate
prompt = "You are a helpful assistant. Answer succinctly.\n\nWhat is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer)
print("Generating...")
_ = model.generate(**inputs, max_new_tokens=64, streamer=streamer)
