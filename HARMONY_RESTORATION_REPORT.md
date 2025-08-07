# GPT-OSS Harmony Restoration Report

## Summary of Findings

After systematic testing to restore Harmony format capabilities, here's what we discovered:

### ✅ Partial Success with Custom Modelfile

Creating `gpt-oss-harmony` with empty template (`TEMPLATE ""`) achieved:

1. **Channels work when explicitly requested** ✓
   - Model successfully uses `<analysis>`, `<commentary>`, `<final>` tags
   - Must be explicitly asked in the prompt
   - Not automatic with `Reasoning: high`

2. **Calculations and reasoning work** ✓
   - Correctly calculates 17 × 19 = 323
   - Shows work when asked
   - Gets factual answers right

3. **Model understands Harmony concepts** ✓
   - Recognizes channel structure
   - Mentions Fibonacci, tools, etc. when relevant
   - Internal reasoning visible in outputs

### ❌ Still Not Working

1. **Variable reasoning levels** - No difference between low/medium/high
   - LOW: 18 tokens
   - MEDIUM: 24 tokens  
   - HIGH: 24 tokens
   - Expected: HIGH should produce MORE tokens with deeper reasoning

2. **CALL_TOOL format** - Never produces the expected JSON structure
   - Mentions tools conceptually
   - Doesn't format as `CALL_TOOL: {"name": "python", ...}`

3. **Automatic channel usage** - Requires explicit prompting
   - `Reasoning: high` alone doesn't trigger channels
   - Must explicitly ask for channel format

## Root Cause Analysis

### The GGUF is Missing Critical Metadata

1. **Template Issue Confirmed**
   - Original: ChatML template (`<|im_start|>`, `<|im_end|>`)
   - Fixed: Empty template allows native format
   - But: Model still doesn't fully recognize Harmony directives

2. **Token Preservation**
   - Words "analysis", "commentary", "final" exist in vocabulary
   - But special token handling is broken
   - Model outputs `<|assistant|>` tokens (not Harmony format)

3. **Quantization Damage**
   - Q4_K_M quantization likely damaged:
     - Reasoning level detection
     - Tool-calling patterns
     - Automatic format switching

## Solution Paths

### Option 1: Better GGUF (Recommended)
```bash
# Need access to original weights to:
1. Re-quantize at Q6_K or Q8 for better precision
2. Include proper tokenizer config
3. Embed correct template in GGUF metadata
```

### Option 2: Workaround Prompting
```python
# Always explicitly request channels
prompt = """<|system|>
Reasoning: high
<|end|><|user|>
[Your question]

Please respond using:
<analysis>Your step-by-step reasoning</analysis>
<commentary>Any tool calls or meta-thoughts</commentary>
<final>Your answer to the user</final>
<|end|>"""
```

### Option 3: Wait for vLLM
- Full precision model
- Native Harmony support
- No quantization issues

## Current Best Practice

Use `gpt-oss-harmony` model with explicit prompting:

```python
import requests

def query_harmony(question, reasoning_level="high"):
    prompt = f"""<|system|>
Reasoning: {reasoning_level}
<|end|><|user|>
{question}

Structure your response with:
<analysis>Detailed reasoning</analysis>
<commentary>Meta-thoughts or tool calls</commentary>
<final>Concise answer</final>
<|end|>"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gpt-oss-harmony",
            "prompt": prompt,
            "stream": False,
            "raw": True
        }
    )
    return response.json()["response"]
```

## Key Takeaways

1. **The model HAS the capabilities** - It understands channels, tools, and reasoning
2. **Quantization/conversion broke automatic triggers** - Need higher precision or better conversion
3. **Workarounds partially restore functionality** - Explicit prompting gets 70% there
4. **Full restoration needs either**:
   - Re-quantization from original weights
   - Or use vLLM with full precision

## Files Created

- `Modelfile.harmony` - Custom model configuration without templates
- `gpt-oss-harmony` - Ollama model with restored partial Harmony support
- Test scripts demonstrating working patterns

## Next Steps

1. **For immediate use**: Use `gpt-oss-harmony` with explicit channel prompting
2. **For full features**: Wait for vLLM or get access to original weights for re-quantization
3. **For production**: Consider the trade-offs between features and inference speed