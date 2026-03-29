# unified-ml: Technical Architecture

## Overview

unified-ml is a custom ML inference engine built specifically for AMD APUs with
Heterogeneous System Architecture (HSA) unified memory. It targets the RDNA 3.5
(gfx1150) GPU found in Ryzen AI 300 series processors.

## The Problem

Current ML frameworks treat GPUs as discrete devices:

```
Standard Framework (PyTorch, ONNX Runtime):

  CPU Memory (DRAM)              GPU Memory (VRAM)
  +------------------+          +------------------+
  | Model Weights    |--copy--->| Model Weights    |
  | Input Tensor     |--copy--->| Input Tensor     |
  | Output Buffer    |<--copy---| Output Buffer    |
  +------------------+          +------------------+
         ^                              ^
         |                              |
    hipMemcpy H2D                  hipMemcpy D2H
```

On an APU, CPU and GPU share the **same physical DRAM**. These copies are
unnecessary -- they move data to the same place it already is.

## The Solution

unified-ml uses HSA unified memory to eliminate all redundant copies:

```
unified-ml on APU:

  Unified Address Space (Shared DRAM)
  +----------------------------------------------+
  | Model Weights  [hipMallocManaged]             |
  | Input Tensor   [hipMallocManaged]             |
  | KV Cache       [hipMallocManaged]             |
  | Output Buffer  [hipMallocManaged]             |
  +----------------------------------------------+
         ^                    ^
         |                    |
     CPU access           GPU access
    (scheduling,        (matmul, attention,
     tokenization,       layer norm, etc.)
     sampling)
```

Both CPU and GPU access the same physical pages. No copies needed.

## Memory Allocation Strategies

We support three allocation strategies with different tradeoffs:

### 1. `hipMallocManaged` (Unified Memory)
- **How it works**: HIP runtime manages page migration automatically
- **Pros**: Simplest API, works transparently
- **Cons**: May trigger page faults on first access from a new processor
- **Best for**: General use, especially when access patterns are predictable

### 2. `hipHostMalloc(Mapped)` (Host-Mapped Memory)
- **How it works**: Allocates in host memory, GPU accesses via IOMMU mapping
- **Pros**: No page migration overhead, deterministic access latency
- **Cons**: GPU accesses may be slightly slower than local VRAM (on discrete GPUs)
- **Best for**: APUs where there is no "local VRAM" -- equivalent performance

### 3. `hipMalloc` + `hipMemcpy` (Standard)
- **How it works**: Separate allocations with explicit copies
- **Pros**: Maximum control, works on all GPUs
- **Cons**: Unnecessary overhead on APUs
- **Best for**: Discrete GPUs only

## Kernel Design

### Matrix Multiplication

The tiled matmul kernel uses shared memory to maximize data reuse:

```
Block (32x32 threads) computes a 32x32 tile of C:

For each K-tile:
  1. Load 32x32 tile of A into shared memory
  2. Load 32x32 tile of B into shared memory
  3. __syncthreads()
  4. Each thread accumulates partial dot product
  5. __syncthreads()
```

On gfx1150 (16 CUs, 2 SIMDs per CU, 32 lanes per SIMD):
- Theoretical peak: ~5.9 TFLOPS FP32 at 2.9 GHz
- Our kernel achieves ~20-40% of peak (comparable to untuned cuBLAS)

### Attention

The attention kernel implements online softmax (numerically stable, single-pass):

```
For each query position q:
  running_max = -inf
  running_sum = 0

  For each K/V tile:
    1. Load K tile into shared memory
    2. Compute scores = Q[q] . K_tile * scale
    3. Update running_max and running_sum
    4. Rescale accumulated output
    5. Add weighted V contributions

  Normalize output by 1/running_sum
```

Key optimization: K and V live in unified memory. The CPU can manage the KV cache
(eviction, rotation, cache compression) while the GPU reads the same data for
attention computation -- no copies needed between cache management and compute.

## Hardware: gfx1150 (RDNA 3.5)

| Feature | Value |
|---|---|
| Architecture | RDNA 3.5 |
| Compute Units | 16 |
| SIMDs per CU | 2 |
| Lanes per SIMD | 32 |
| Max Clock | 2900 MHz |
| L2 Cache | 2 MB |
| Memory | Shared system DRAM (LPDDR5x) |
| Memory Bandwidth | ~120 GB/s (shared with CPU) |
| FP32 Peak | ~5.9 TFLOPS |

## Roadmap

### Phase 1 (Current): Proof of Concept
- [x] Unified memory benchmark (hipMallocManaged vs hipMalloc+hipMemcpy)
- [x] Tiled FP32 matmul kernel
- [x] Scaled dot-product attention kernel
- [x] Correctness verification against CPU reference
- [x] Python wrapper for benchmarks

### Phase 2: Inference Engine
- [ ] Weight loading (safetensors, GGUF)
- [ ] Layer normalization kernel
- [ ] RoPE (rotary position embedding) kernel
- [ ] SiLU/GELU activation kernels
- [ ] Full transformer decoder block
- [ ] KV cache management in unified memory

### Phase 3: Optimization
- [ ] FP16 / BF16 support (RDNA 3.5 has native FP16)
- [ ] Wave32/Wave64 occupancy tuning
- [ ] Async compute overlap (CPU tokenization while GPU computes)
- [ ] Memory bandwidth optimization (coalescing, padding)
- [ ] Profile-guided tuning for specific models

### Phase 4: Integration
- [ ] XDNA NPU offload for INT8 quantized layers
- [ ] Python model loading API
- [ ] Compatible with HuggingFace model format
- [ ] Benchmarks against llama.cpp, vLLM
