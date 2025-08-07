# GPT-OSS Ollama Comprehensive Test Report

## Executive Summary

Comprehensive testing of GPT-OSS model (gpt-oss:latest) using Ollama was conducted across 8 major capability areas. The model demonstrates strong factual accuracy but shows unexpected behavior in some advanced features.

**Test Date:** 2025-08-07  
**Model:** gpt-oss:latest (13 GB GGUF)  
**Platform:** Ollama

---

## Test Results Summary

### 1. Variable-Effort Reasoning ❌ Unexpected Behavior

The model does NOT exhibit the expected scaling behavior with reasoning effort levels:

- **LOW effort:** 343 tokens, 74.2 tok/s
- **MEDIUM effort:** 288 tokens, 78.5 tok/s  
- **HIGH effort:** 276 tokens, 78.8 tok/s

**Finding:** Token generation DECREASES with higher reasoning levels (opposite of expected)
- The model got the wrong answer (210 instead of 300) for the train problem
- No clear chain-of-thought scaling observed

### 2. Harmony Chat Format ❌ Not Supported

The model does NOT use the Harmony format channels (<analysis>, <commentary>, <final>):

- **Channel usage:** 0/3 tests used any channels
- **Response pattern:** Model produces gibberish or ignores channel instructions
- Model appears to use internal tokens like `<|end|><|start|>assistant<|channel|>`

### 3. Tool Calling ⚠️ Partial Support

The model understands tool-related queries but doesn't use formal tool calling:

- **CALL_TOOL pattern:** 0/3 tests (0%)
- **Code present:** 2/3 tests (67%)
- **Content accuracy:** 17/20 keywords found (85%)

**Finding:** Model can solve problems requiring tools but doesn't follow tool-calling format

### 4. Reasoning-Length vs Accuracy ✅ Good Performance

Simple math problems show high accuracy across all reasoning levels:

- **Accuracy:** 100% on simple percentage calculation
- **Token usage:** Varies but doesn't correlate with reasoning level
- Some errors occurred with None token counts in complex tests

### 5. Long Context Handling ✅ Excellent

The model successfully handles long contexts:

- **Main test:** 2/2 questions answered correctly (100%)
- **Retrieved secret code** from 11K character document
- **Context scaling:** Successfully processed 1K, 5K, and 10K character contexts
- **Processing time:** Scales linearly with context size

### 6. Safety & Jailbreak Resistance ⚠️ Mixed Results

- **Overall compliance:** 3/5 tests (60%)
- **Educational security:** 2/3 appropriate responses
- **System probe:** 1/1 correctly deflected
- **Instruction override:** 0/1 failed (followed override)

**Finding:** Model has basic safety but can be overridden with simple prompts

### 7. Hallucination Baseline ✅ Excellent

Extremely low hallucination rate on factual questions:

- **LOW reasoning:** 100% accuracy (8/8)
- **MEDIUM reasoning:** 87.5% accuracy (7/8)
- **HIGH reasoning:** 100% accuracy (8/8)

**Finding:** Model is highly reliable for factual information

### 8. Performance Metrics

Average performance across tests:
- **Token generation speed:** 70-80 tokens/second
- **Response time:** 2-6 seconds for most queries
- **Long context:** 25-30 seconds for very long prompts

---

## Key Findings

### ✅ Strengths
1. **Excellent factual accuracy** - Near-zero hallucination rate
2. **Strong long-context handling** - Successfully processes 10K+ character contexts
3. **Good problem-solving** - Correctly solves math and logic problems
4. **Fast inference** - 70-80 tok/s on consumer hardware

### ❌ Limitations
1. **Variable-effort reasoning not working** - No chain-of-thought scaling observed
2. **Harmony format not supported** - Channels are ignored or produce errors
3. **No formal tool calling** - Doesn't follow structured tool-call format
4. **Weak instruction following** - Can be easily overridden
5. **Response artifacts** - Contains internal tokens and formatting issues

### ⚠️ Unexpected Behaviors
1. Model produces internal tokens like `<|end|><|start|>assistant<|channel|>`
2. Reasoning levels produce INVERSE token counts (high = fewer tokens)
3. Frequent gibberish patterns in responses (`.....???...`)
4. Math errors on complex problems despite correct methodology

---

## Recommendations

1. **For Production Use:**
   - ✅ Use for factual Q&A and information retrieval
   - ✅ Use for long-document analysis
   - ❌ Don't rely on variable-effort reasoning
   - ❌ Don't use Harmony format or structured tool calling

2. **Model Configuration:**
   - Keep temperature low (0.1-0.3) for factual tasks
   - Use "Reasoning: high" in system prompt for best results
   - Expect 70-80 tok/s performance

3. **Further Testing Needed:**
   - Test with different quantization levels
   - Compare with full-precision model
   - Test with vLLM for better format support

---

## Test Files Generated

- `variable_effort_results.json` - Variable reasoning test data
- `harmony_format_results.json` - Channel format test data
- `tool_calls_results.json` - Tool calling test data
- `reasoning_simple_results.json` - Simple reasoning accuracy
- `long_context_results.json` - Context handling test data
- `safety_results.json` - Safety evaluation data
- `hallucination_results.json` - Factual accuracy data

---

## Conclusion

GPT-OSS on Ollama performs well for general Q&A and factual tasks but lacks support for advanced features described in the technical report. The GGUF quantization or Ollama implementation may be limiting the model's advanced capabilities. Consider testing with vLLM or full-precision weights for complete feature support.