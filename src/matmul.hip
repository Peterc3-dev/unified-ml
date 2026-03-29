#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>

// ============================================================================
// unified-ml: Optimized FP32 Matrix Multiply for gfx1150 (RDNA 3.5)
// Uses shared memory tiling + unified memory for zero-copy host access
// ============================================================================

#define TILE_DIM 32

// ============================================================================
// Tiled matmul kernel using shared memory
// A[M,K] * B[K,N] = C[M,N]
// Each thread computes one element of C by tiling over the K dimension
// ============================================================================
__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; t++) {
        // Load A tile
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Naive matmul for comparison
// ============================================================================
__global__ void naive_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Benchmark helper
// ============================================================================
typedef void (*matmul_fn)(const float*, const float*, float*, int, int, int);

double benchmark_matmul(matmul_fn kernel_launcher, int M, int N, int K,
                        int iterations, const char* label) {
    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    float *A, *B, *C;
    hipMallocManaged(&A, a_bytes);
    hipMallocManaged(&B, b_bytes);
    hipMallocManaged(&C, c_bytes);

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = sinf((float)i * 0.001f);
    for (int i = 0; i < K * N; i++) B[i] = cosf((float)i * 0.001f);

    int device;
    hipGetDevice(&device);
    hipMemPrefetchAsync(A, a_bytes, device, 0);
    hipMemPrefetchAsync(B, b_bytes, device, 0);
    hipDeviceSynchronize();

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Warmup
    hipLaunchKernelGGL(kernel_launcher, grid, block, 0, 0, A, B, C, M, N, K);
    hipDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(kernel_launcher, grid, block, 0, 0, A, B, C, M, N, K);
        hipDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count()
                / iterations;

    // Compute GFLOPS: 2*M*N*K operations for matmul
    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    printf("  %-8s  [%4d x %4d] x [%4d x %4d]  %8.2f ms  %8.2f GFLOPS\n",
           label, M, K, K, N, ms, gflops);

    hipFree(A); hipFree(B); hipFree(C);
    return ms;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("===========================================================\n");
    printf("  unified-ml: Tiled FP32 MatMul for gfx1150\n");
    printf("===========================================================\n\n");

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("GPU:           %s\n", props.name);
    printf("Architecture:  %s\n", props.gcnArchName);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("Max Clock:     %d MHz\n", props.clockRate / 1000);

    // Theoretical peak: CUs * SIMDs/CU * lanes * 2 (FMA) * clock
    // gfx1150: 16 CU * 2 SIMD * 32 lanes * 2 * 2.9 GHz
    double peak_fp32 = 16.0 * 2.0 * 32.0 * 2.0 * (props.clockRate / 1e6);
    printf("Theo. Peak:    %.1f GFLOPS (FP32, FMA)\n\n", peak_fp32);

    // Correctness check
    printf("--- Correctness Check ---\n");
    {
        int N = 128;
        size_t bytes = (size_t)N * N * sizeof(float);
        float *A, *B, *C_tiled, *C_naive;
        hipMallocManaged(&A, bytes);
        hipMallocManaged(&B, bytes);
        hipMallocManaged(&C_tiled, bytes);
        hipMallocManaged(&C_naive, bytes);

        for (int i = 0; i < N * N; i++) {
            A[i] = sinf((float)i * 0.001f);
            B[i] = cosf((float)i * 0.001f);
        }

        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

        hipLaunchKernelGGL(tiled_matmul_kernel, grid, block, 0, 0,
                           A, B, C_tiled, N, N, N);
        hipLaunchKernelGGL(naive_matmul_kernel, grid, block, 0, 0,
                           A, B, C_naive, N, N, N);
        hipDeviceSynchronize();

        float max_err = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float err = fabsf(C_tiled[i] - C_naive[i]);
            if (err > max_err) max_err = err;
        }
        printf("  Tiled vs Naive (N=%d): max error = %.6e %s\n\n",
               N, max_err, (max_err < 1e-3f) ? "[PASS]" : "[FAIL]");

        hipFree(A); hipFree(B); hipFree(C_tiled); hipFree(C_naive);
    }

    // Benchmarks
    printf("--- Performance Comparison ---\n\n");
    printf("  %-8s  %-28s  %10s  %12s\n",
           "Kernel", "Dimensions", "Time", "Throughput");
    printf("  %-8s  %-28s  %10s  %12s\n",
           "--------", "----------------------------", "----------", "------------");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    for (int s = 0; s < 5; s++) {
        int N = sizes[s];
        int iters = (N <= 512) ? 20 : (N <= 2048) ? 5 : 2;
        benchmark_matmul(naive_matmul_kernel, N, N, N, iters, "Naive");
        benchmark_matmul(tiled_matmul_kernel, N, N, N, iters, "Tiled");
        printf("\n");
    }

    printf("--- Notes ---\n");
    printf("All allocations use hipMallocManaged (unified memory).\n");
    printf("Tiled kernel uses %dx%d shared memory tiles.\n", TILE_DIM, TILE_DIM);
    printf("GFLOPS = 2*M*N*K / (time_ms * 1e6)\n");

    return 0;
}
