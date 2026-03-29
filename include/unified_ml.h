#ifndef UNIFIED_ML_H
#define UNIFIED_ML_H

// ============================================================================
// unified-ml: Public API Header
// ML Inference Engine for AMD APU Unified Memory Architecture
// Target: RDNA 3.5 (gfx1150) -- Radeon 890M / 880M
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// ============================================================================
// Error codes
// ============================================================================
typedef enum {
    UML_SUCCESS = 0,
    UML_ERROR_HIP = -1,
    UML_ERROR_INVALID_ARG = -2,
    UML_ERROR_OUT_OF_MEMORY = -3,
    UML_ERROR_UNSUPPORTED = -4,
} uml_status_t;

// ============================================================================
// Memory allocation strategies
// ============================================================================
typedef enum {
    UML_MEM_UNIFIED,       // hipMallocManaged -- best for APU
    UML_MEM_HOST_MAPPED,   // hipHostMalloc(Mapped) -- alternative zero-copy
    UML_MEM_DEVICE,        // hipMalloc -- discrete-GPU style (for comparison)
    UML_MEM_PINNED,        // hipHostMalloc(Default) -- pinned host memory
} uml_mem_strategy_t;

// ============================================================================
// Tensor descriptor
// ============================================================================
typedef struct {
    float* data;              // Pointer to data (unified address space)
    int    ndim;              // Number of dimensions
    int    shape[4];          // Shape (max 4D: batch, seq, heads, dim)
    size_t stride[4];         // Strides in elements
    size_t num_elements;      // Total number of elements
    uml_mem_strategy_t mem;   // Memory strategy used for allocation
} uml_tensor_t;

// ============================================================================
// Device info
// ============================================================================
typedef struct {
    char   name[256];
    char   arch[64];
    int    compute_units;
    int    max_clock_mhz;
    size_t total_memory_mb;
    int    l2_cache_kb;
    int    supports_managed_memory;
    int    supports_concurrent_managed;
    int    is_apu;               // 1 if integrated GPU (APU)
} uml_device_info_t;

// ============================================================================
// Core API
// ============================================================================

// Initialize the runtime (must be called first)
uml_status_t uml_init(void);

// Shut down and release all resources
uml_status_t uml_shutdown(void);

// Get device information
uml_status_t uml_get_device_info(uml_device_info_t* info);

// ============================================================================
// Tensor operations
// ============================================================================

// Allocate a tensor with the specified memory strategy
uml_status_t uml_tensor_alloc(uml_tensor_t* tensor, int ndim, const int* shape,
                               uml_mem_strategy_t mem);

// Free a tensor
uml_status_t uml_tensor_free(uml_tensor_t* tensor);

// Prefetch tensor data to GPU (hint for unified memory)
uml_status_t uml_tensor_prefetch_gpu(uml_tensor_t* tensor);

// Prefetch tensor data to CPU
uml_status_t uml_tensor_prefetch_cpu(uml_tensor_t* tensor);

// ============================================================================
// Compute operations
// ============================================================================

// Matrix multiply: C = A * B
// A[M,K] * B[K,N] -> C[M,N]
uml_status_t uml_matmul(const uml_tensor_t* A, const uml_tensor_t* B,
                         uml_tensor_t* C);

// Scaled dot-product attention
// Q[seq,dim] * K[seq,dim]^T -> scores -> softmax -> * V[seq,dim] -> out[seq,dim]
uml_status_t uml_attention(const uml_tensor_t* Q, const uml_tensor_t* K,
                            const uml_tensor_t* V, uml_tensor_t* output,
                            float scale);

// ============================================================================
// Benchmark utilities
// ============================================================================

// Run the unified memory benchmark suite, print results to stdout
uml_status_t uml_benchmark_memory(void);

// Run the matmul benchmark suite
uml_status_t uml_benchmark_matmul(void);

// Run the attention benchmark suite
uml_status_t uml_benchmark_attention(void);

#ifdef __cplusplus
}
#endif

#endif // UNIFIED_ML_H
