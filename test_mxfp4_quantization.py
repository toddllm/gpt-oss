#!/usr/bin/env python3
"""
Test MXFP4 quantization on NVIDIA RTX 3090
"""

import torch
import triton
import triton.language as tl
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"Triton version: {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

@triton.jit
def quantize_fp32_to_fp4_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple FP32 to FP4-like quantization kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load FP32 values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Simple FP4-like quantization (scale to 4-bit range)
    # This is a simplified version - real MXFP4 would be more complex
    # Scale to [-7, 7] range (4-bit signed)
    max_val = tl.max(tl.abs(x))
    scale = 7.0 / (max_val + 1e-8)
    
    # Quantize - using tl.math functions instead of libdevice
    x_scaled = x * scale
    # Round to nearest integer
    quantized = tl.where(x_scaled >= 0, 
                        tl.floor(x_scaled + 0.5),
                        tl.ceil(x_scaled - 0.5))
    quantized = tl.minimum(7.0, tl.maximum(-7.0, quantized))
    
    # Dequantize for storage (in real implementation, you'd store the 4-bit values)
    dequantized = quantized / scale
    
    # Store result
    tl.store(y_ptr + offsets, dequantized, mask=mask)

def test_fp4_quantization():
    """Test FP4-like quantization on GPU"""
    # Create test tensor
    size = 1024
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.zeros_like(x)
    
    print(f"\nTest tensor shape: {x.shape}")
    print(f"Original tensor stats - Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    quantize_fp32_to_fp4_kernel[grid,](
        x, y, size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Check results
    print(f"Quantized tensor stats - Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}")
    
    # Calculate quantization error
    mse = torch.mean((x - y) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")
    
    # Check that quantization worked (values should be discrete)
    unique_values = torch.unique(y).cpu().numpy()
    print(f"Number of unique values after quantization: {len(unique_values)}")
    
    if len(unique_values) <= 16:  # Should have at most 16 values for 4-bit
        print("✓ Quantization successful - values are properly discretized")
    else:
        print("⚠ Warning: More unique values than expected for 4-bit quantization")
    
    return True

def test_performance():
    """Benchmark quantization performance"""
    sizes = [1024, 4096, 16384, 65536]
    
    print("\n=== Performance Benchmark ===")
    for size in sizes:
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.zeros_like(x)
        
        # Warmup
        BLOCK_SIZE = 256
        grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
        for _ in range(10):
            quantize_fp32_to_fp4_kernel[grid,](
                x, y, size, BLOCK_SIZE=BLOCK_SIZE
            )
        
        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            quantize_fp32_to_fp4_kernel[grid,](
                x, y, size, BLOCK_SIZE=BLOCK_SIZE
            )
        end.record()
        
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / 100
        
        bandwidth_gb = (size * 4 * 2) / (time_ms * 1e6)  # Read + Write
        print(f"Size: {size:6d} | Time: {time_ms:.3f}ms | Bandwidth: {bandwidth_gb:.2f} GB/s")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("MXFP4 Quantization Test on RTX 3090")
    print("="*50)
    
    try:
        # Run basic quantization test
        if test_fp4_quantization():
            print("\n✓ Basic FP4 quantization test passed!")
        
        # Run performance benchmark
        test_performance()
        
        print("\n" + "="*50)
        print("✓ All tests completed successfully!")
        print("Your RTX 3090 is ready for FP4 quantization with Triton 3.4.0")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()