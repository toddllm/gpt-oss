# GPT-OSS Ollama Troubleshooting Report

## Key Findings

After systematic troubleshooting, I've identified the root causes of why GPT-OSS advanced features aren't working properly on Ollama:

### 🔴 Primary Issue: Wrong Chat Template

**The model is configured with ChatML template instead of native Harmony format:**
```
TEMPLATE "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"
```

**Expected Harmony template:**
```
<|system|>
{{ .System }}
<|end|><|user|>
{{ .Prompt }}
<|end|>
```

### ✅ Partial Success: Channels Work When Explicitly Requested

When explicitly asked to use channels in the prompt, the model DOES respond with proper formatting:
- ✅ `<analysis>`, `<commentary>`, `<final>` tags work
- ✅ Model understands the channel concept
- ❌ But doesn't use them automatically with `Reasoning: high`

### ⚠️ Variable Reasoning: Inverse Behavior

The `Reasoning: low/medium/high` directive shows **opposite** behavior:
- **HIGH reasoning**: 49-276 tokens (shortest)
- **MEDIUM reasoning**: 55-288 tokens 
- **LOW reasoning**: 163-1024 tokens (longest)

This suggests the quantization or template is interfering with the reasoning control mechanism.

### 🟡 Tool Calling: Partially Functional

The model shows understanding of tools but inconsistent formatting:
- ✅ Mentions tools and understands when to use them
- ✅ Sometimes produces `CALL_TOOL` format (especially in commentary channel)
- ❌ Doesn't consistently follow the JSON schema
- ❌ Often solves problems directly instead of calling tools

## Root Causes Identified

### 1. **Template Mismatch**
   - Ollama applies ChatML template (`<|im_start|>`, `<|im_end|>`)
   - Model expects Harmony format (`<|system|>`, `<|end|>`)
   - This breaks the native instruction following

### 2. **Token Preservation Issues**
   - Special tokens like `<|channel|>` appear in outputs
   - Suggests tokens are present but not properly handled
   - The model outputs internal control tokens: `<|end|><|start|>assistant<|channel|>`

### 3. **Quantization Effects**
   - Q4_K_M quantization may lose precision for:
     - Reasoning level control
     - Format instruction following
     - Tool detection patterns

## Solutions & Workarounds

### Immediate Workarounds for Ollama Users:

1. **Explicit Channel Instructions**
   ```python
   prompt = """Please respond using this format:
   <analysis>Your reasoning here</analysis>
   <commentary>Any comments or tool calls</commentary>
   <final>Your answer</final>
   
   Question: [your question]"""
   ```

2. **Direct API Usage** (bypasses some template issues)
   ```python
   import requests
   
   payload = {
       "model": "gpt-oss:latest",
       "prompt": your_prompt,
       "stream": False,
       "raw": True  # Helps but doesn't fully solve
   }
   response = requests.post("http://localhost:11434/api/generate", json=payload)
   ```

3. **Tool Calling Prompt Engineering**
   ```python
   # Include tools in prompt and explicitly ask for CALL_TOOL format
   prompt = f"""Tools: {json.dumps(tools)}
   
   When you need a tool, respond with:
   <commentary>
   CALL_TOOL: {{"name": "tool_name", "arguments": {{...}}}}
   </commentary>
   
   User: {question}"""
   ```

### Proper Fix Options:

1. **Create Custom Modelfile**
   ```dockerfile
   FROM gpt-oss:latest
   TEMPLATE "<|system|>
   {{ .System }}
   <|end|><|user|>
   {{ .Prompt }}
   <|end|><|assistant|>"
   PARAMETER stop <|end|>
   PARAMETER stop <|user|>
   ```

2. **Re-quantize with Proper Config**
   - Use higher quantization (Q6_K or Q8)
   - Preserve special tokens during conversion
   - Include proper chat template in GGUF metadata

3. **Use vLLM Instead**
   - Full precision model
   - Native Harmony format support
   - Proper handling of all special tokens

## Test Results Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Variable Reasoning | ❌ | Inverse behavior (high = fewer tokens) |
| Harmony Channels | 🟡 | Works when explicitly requested |
| Tool Calling | 🟡 | Understands but inconsistent format |
| Long Context | ✅ | Works well up to 10K+ chars |
| Factual Accuracy | ✅ | Near 100% accuracy |
| Safety | ✅ | Basic safety preserved |

## Recommendations

1. **For Ollama users wanting advanced features:**
   - Wait for proper GGUF with correct template
   - Use explicit prompting workarounds
   - Consider vLLM for full capabilities

2. **For model maintainers:**
   - Release GGUF with native Harmony template
   - Document template requirements clearly
   - Test quantized versions for feature preservation

3. **For production use:**
   - Use vLLM with full precision for complete features
   - Ollama/GGUF suitable for basic Q&A only
   - Implement explicit prompt engineering for channels/tools

## Conclusion

The GPT-OSS model has the capabilities described in the paper, but they're being blocked by:
1. Wrong chat template in Ollama configuration
2. Quantization effects on control mechanisms
3. Token handling issues in the GGUF format

The model works well for general tasks but requires vLLM or proper reconfiguration to access advanced Harmony features.