from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "openai/gpt-oss-20b"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
out = model.generate(**tok("How big is the Sun?", return_tensors="pt").to(0),
                     max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))

