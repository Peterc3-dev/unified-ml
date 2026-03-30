# unified-ml

**A custom ML inference engine optimized for AMD APU unified memory architecture.**

Built for AMD Ryzen AI 300 series (Strix Point) processors with Radeon 890M/880M integrated GPUs (RDNA 3.5, gfx1150).

## Why?

Current ML frameworks (PyTorch, ONNX Runtime) treat GPUs as discrete devices with separate memory pools. On APUs, CPU and GPU share the same physical memory -- but existing frameworks still copy data between "host" and "device" as if they were separate.

unified-ml eliminates this overhead by using HSA unified memory primitives, enabling:

- **Zero-copy inference** -- no CPU<->GPU data transfers
- **Shared KV-cache** -- attention cache lives in unified memory, accessible by both CPU and GPU
- **FP32-native compute** -- optimized for RDNA 3.5's compute units
- **Custom attention kernel** -- tiled, online-softmax attention built for gfx1150

## Quick Start

```bash
# Build (requires ROCm 7.x with HIP)
cd ~/projects/unified-ml
mkdir -p build
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx1150 src/unified_memory.hip -o build/unified_benchmark
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx1150 src/matmul.hip -o build/matmul_benchmark
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx1150 src/attention.hip -o build/attention_benchmark

# Run benchmarks
./build/unified_benchmark
./build/matmul_benchmark
./build/attention_benchmark

# Python runner
python python/unified_ml.py --benchmark all
```

## Architecture

On APUs, `hipMallocManaged()` allocates memory in a unified address space shared by CPU and GPU. This eliminates the `hipMalloc()` + `hipMemcpy()` pattern that discrete-GPU frameworks use:

```
Traditional (discrete GPU mindset):     unified-ml (APU-native):
  CPU RAM  --hipMemcpy-->  GPU VRAM       Shared DRAM
  [weights]                [weights]      [weights] <-- both CPU and GPU
  [input]                  [input]        [input]       read/write directly
  [output]  <--hipMemcpy-- [output]       [output]
```

See [docs/architecture.md](docs/architecture.md) for the full technical design.

## Benchmarks

### Vulkan compute (IREE)

| Backend | GFLOPS | Notes |
|---------|--------|-------|
| **Vulkan (IREE)** | **1,084.8** | Full VRAM + GTT access via `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` |
| ROCm (HIP) | ~680 | Limited to BIOS VRAM carveout (`hipMallocManaged` broken on gfx1150) |

**Vulkan outperforms ROCm by ~60% on the Radeon 890M** because Vulkan correctly addresses both VRAM and GTT memory (80+ GB total on a 32 GB system), while ROCm's `hipMallocManaged` is broken on gfx1150 and sees only the BIOS-configured VRAM carveout.

These kernels are used by [R.A.G-Race-Router](https://github.com/Peterc3-dev/rag-race-router), the tri-processor inference runtime that routes GPU work through Vulkan instead of ROCm.

### Memory strategy comparison

The PoC benchmark compares three memory strategies on the same matmul kernel:

| Strategy | Description | Copy Overhead |
|---|---|---|
| Standard (`hipMalloc` + `hipMemcpy`) | Discrete-GPU style | Full copy each iteration |
| Unified (`hipMallocManaged`) | HSA managed memory | Zero copies |
| Host-Mapped (`hipHostMalloc`) | Pinned + mapped | Zero copies |

## Project Structure

```
unified-ml/
  src/
    unified_memory.hip   -- Memory strategy benchmark (standard vs unified vs mapped)
    matmul.hip           -- Tiled FP32 matmul kernel (naive vs shared-memory tiled)
    attention.hip        -- Scaled dot-product attention with online softmax
  include/
    unified_ml.h         -- Public C API header
  python/
    unified_ml.py        -- Python wrapper (subprocess-based)
  benchmarks/
    pytorch_comparison.py -- Compare against PyTorch on the same operations
  docs/
    architecture.md      -- Full technical architecture document
```

## Supported Hardware

- AMD Ryzen AI 9 HX 370 / Radeon 890M (gfx1150)
- AMD Ryzen AI 7 PRO 360 / Radeon 880M (gfx1150)
- Any RDNA 3.5 APU with HSA unified memory

## Requirements

- ROCm 7.x (tested with 7.2.0)
- HIP compiler (`/opt/rocm/bin/hipcc`)
- Linux (tested on CachyOS)

## Status

Proof of concept -- demonstrating the unified memory advantage for ML inference workloads on AMD APUs.

## Roadmap

- [x] Unified memory PoC benchmark
- [x] Tiled FP32 matmul kernel
- [x] Custom attention kernel for gfx1150
- [x] Correctness verification (GPU vs CPU reference)
- [x] Python benchmark wrapper
- [x] Vulkan compute backend (Kompute)
- [x] Vulkan vs HIP benchmark comparison
- [ ] Full transformer inference engine
- [ ] Model loading (GGUF / safetensors)
- [ ] FP16/BF16 kernels
- [ ] KV cache management
- [x] NPU proof of life (amdxdna driver + /dev/accel/accel0)
- [ ] XDNA NPU integration (IREE runtime)
- [ ] Python inference API

## License

MIT
