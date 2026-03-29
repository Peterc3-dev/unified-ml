#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>

// ============================================================================
// unified-ml: Unified Memory ML Inference Engine for AMD APUs
// Proof of Concept - Memory Benchmark
// Target: gfx1150 (RDNA 3.5, Radeon 890M)
// ============================================================================

// Naive matmul kernel (compute-bound baseline)
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Helper: check HIP errors
#define HIP_CHECK(call)                                                       \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    hipGetErrorString(err));                                    \
            return -1.0;                                                       \
        }                                                                      \
    } while (0)

// ============================================================================
// Benchmark 1: Standard HIP (discrete-GPU style: alloc + copy)
// ============================================================================
double benchmark_standard(int N, int iterations) {
    size_t bytes = (size_t)N * N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed for N=%d\n", N);
        free(h_A); free(h_B); free(h_C);
        return -1.0;
    }

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i + 50) % 100) / 100.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warmup
    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, d_A, d_B, d_C, N);
    hipDeviceSynchronize();

    // Timed loop: includes copy-in, compute, copy-out
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, d_A, d_B, d_C, N);
        hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count()
                / iterations;

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return ms;
}

// ============================================================================
// Benchmark 2: Unified memory (zero-copy on APU)
// ============================================================================
double benchmark_unified(int N, int iterations) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A, *B, *C;
    HIP_CHECK(hipMallocManaged(&A, bytes));
    HIP_CHECK(hipMallocManaged(&B, bytes));
    HIP_CHECK(hipMallocManaged(&C, bytes));

    // Initialize on CPU -- no copy needed, same physical memory
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i + 50) % 100) / 100.0f;
    }

    // Prefetch hint to GPU (optional on APU, but good practice)
    int device;
    hipGetDevice(&device);
    hipMemPrefetchAsync(A, bytes, device, 0);
    hipMemPrefetchAsync(B, bytes, device, 0);
    hipDeviceSynchronize();

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warmup
    hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, A, B, C, N);
    hipDeviceSynchronize();

    // Timed loop: no memcpy at all -- GPU reads/writes unified memory directly
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, A, B, C, N);
        hipDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count()
                / iterations;

    hipFree(A); hipFree(B); hipFree(C);
    return ms;
}

// ============================================================================
// Benchmark 3: hipHostMalloc with mapped flag (alternative zero-copy)
// ============================================================================
double benchmark_host_mapped(int N, int iterations) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A, *B, *C;
    HIP_CHECK(hipHostMalloc(&A, bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&B, bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&C, bytes, hipHostMallocMapped));

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i + 50) % 100) / 100.0f;
    }

    // Get device pointers for the host-mapped memory
    float *dA, *dB, *dC;
    HIP_CHECK(hipHostGetDevicePointer((void**)&dA, A, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&dB, B, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&dC, C, 0));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warmup
    hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, dA, dB, dC, N);
    hipDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, dA, dB, dC, N);
        hipDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count()
                / iterations;

    hipHostFree(A); hipHostFree(B); hipHostFree(C);
    return ms;
}

// ============================================================================
// Correctness verification
// ============================================================================
bool verify_result(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *A, *B, *C_gpu, *C_cpu;
    hipMallocManaged(&A, bytes);
    hipMallocManaged(&B, bytes);
    hipMallocManaged(&C_gpu, bytes);
    C_cpu = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i + 50) % 100) / 100.0f;
    }

    // GPU matmul
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, A, B, C_gpu, N);
    hipDeviceSynchronize();

    // CPU reference
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C_cpu[row * N + col] = sum;
        }
    }

    // Compare
    float max_err = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("  Verification (N=%d): max absolute error = %.6e %s\n",
           N, max_err, (max_err < 1e-2f) ? "[PASS]" : "[FAIL]");

    hipFree(A); hipFree(B); hipFree(C_gpu);
    free(C_cpu);
    return max_err < 1e-2f;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("===========================================================\n");
    printf("  unified-ml: Unified Memory vs Standard HIP on AMD APU\n");
    printf("===========================================================\n\n");

    // Device info
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("GPU:             %s\n", props.name);
    printf("Architecture:    %s\n", props.gcnArchName);
    printf("Compute Units:   %d\n", props.multiProcessorCount);
    printf("Max Clock:       %d MHz\n", props.clockRate / 1000);
    printf("Memory:          %zu MB (shared with CPU)\n",
           props.totalGlobalMem / (1024 * 1024));
    printf("L2 Cache:        %d KB\n", props.l2CacheSize / 1024);
    printf("Managed Memory:  %s\n", props.managedMemory ? "YES" : "NO");
    printf("Concurrent Managed Access: %s\n",
           props.concurrentManagedAccess ? "YES" : "NO");
    printf("\n");

    // Correctness
    printf("--- Correctness Check ---\n");
    verify_result(64);
    verify_result(256);
    printf("\n");

    // Benchmark table
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = 4;

    printf("--- Memory Strategy Benchmark ---\n\n");
    printf("%-6s  %-14s  %-14s  %-14s  %-12s  %-12s\n",
           "N", "Standard (ms)", "Unified (ms)", "HostMap (ms)",
           "Uni Speedup", "Map Speedup");
    printf("%-6s  %-14s  %-14s  %-14s  %-12s  %-12s\n",
           "------", "--------------", "--------------", "--------------",
           "------------", "------------");

    for (int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        int iters = (N <= 512) ? 20 : 5;

        double std_ms  = benchmark_standard(N, iters);
        double uni_ms  = benchmark_unified(N, iters);
        double map_ms  = benchmark_host_mapped(N, iters);

        if (std_ms < 0 || uni_ms < 0 || map_ms < 0) {
            printf("%-6d  ERROR in one or more benchmarks\n", N);
            continue;
        }

        double uni_speedup = std_ms / uni_ms;
        double map_speedup = std_ms / map_ms;

        printf("%-6d  %-14.2f  %-14.2f  %-14.2f  %-12.2fx  %-12.2fx\n",
               N, std_ms, uni_ms, map_ms, uni_speedup, map_speedup);
    }

    printf("\n--- Analysis ---\n");
    printf("On APUs with unified memory (HSA), hipMallocManaged and\n");
    printf("hipHostMalloc(Mapped) eliminate all CPU<->GPU memory copies.\n\n");
    printf("The speedup comes from:\n");
    printf("  1. Zero-copy data access (same physical DRAM)\n");
    printf("  2. No PCIe bottleneck (integrated GPU, no bus transfer)\n");
    printf("  3. CPU can directly read GPU results without memcpy\n");
    printf("  4. Reduced API overhead from eliminating hipMemcpy calls\n\n");
    printf("For ML inference on APUs, unified memory allows:\n");
    printf("  - KV-cache shared between CPU scheduling and GPU compute\n");
    printf("  - Weights loaded once, accessible by both processors\n");
    printf("  - Token-by-token decode without copy overhead\n");

    return 0;
}
