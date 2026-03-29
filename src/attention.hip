#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>

// ============================================================================
// unified-ml: Custom Attention Kernel for gfx1150 (RDNA 3.5)
// Optimized for unified memory -- KV-cache lives in shared address space
// ============================================================================

#define TILE_SIZE 32

// ============================================================================
// Kernel: Scaled Dot-Product Attention (single-head, single-threaded per query)
//
//   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// Each block processes one query position with a single thread.
// This is a correctness-first implementation; the parallelism comes from
// processing many query positions in parallel (one block per query).
//
// For a PoC this is fine -- it proves unified memory works for attention.
// A production kernel would parallelize across the head_dim and use
// warp-level reductions for the softmax.
// ============================================================================
__global__ void scaled_dot_product_attention(
    const float* __restrict__ Q,      // [seq_len, head_dim]
    const float* __restrict__ K,      // [seq_len, head_dim]
    const float* __restrict__ V,      // [seq_len, head_dim]
    float* __restrict__ output,       // [seq_len, head_dim]
    int seq_len,
    int head_dim,
    float scale
) {
    int q_pos = blockIdx.x;
    if (q_pos >= seq_len) return;

    // Only thread 0 does the work (one thread per query position)
    if (threadIdx.x != 0) return;

    const float* q_row = Q + q_pos * head_dim;

    // Online softmax: single pass over K/V
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Temporary output accumulator (stack, up to head_dim=128)
    float out_accum[128];
    for (int d = 0; d < head_dim; d++) {
        out_accum[d] = 0.0f;
    }

    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        // Compute Q[q_pos] . K[k_pos] * scale
        float dot = 0.0f;
        const float* k_row = K + k_pos * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        float score = dot * scale;

        // Online softmax update
        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float exp_score = expf(score - new_max);

        // Rescale existing accumulator and add new contribution
        running_sum = running_sum * correction + exp_score;
        const float* v_row = V + k_pos * head_dim;
        for (int d = 0; d < head_dim; d++) {
            out_accum[d] = out_accum[d] * correction + exp_score * v_row[d];
        }
        running_max = new_max;
    }

    // Normalize and write output
    float inv_sum = 1.0f / running_sum;
    float* out_row = output + q_pos * head_dim;
    for (int d = 0; d < head_dim; d++) {
        out_row[d] = out_accum[d] * inv_sum;
    }
}

// ============================================================================
// CPU reference implementation for verification
// ============================================================================
void attention_cpu(const float* Q, const float* K, const float* V,
                   float* output, int seq_len, int head_dim, float scale) {
    float* scores = new float[seq_len];

    for (int q = 0; q < seq_len; q++) {
        // Compute scores
        float max_score = -1e30f;
        for (int k = 0; k < seq_len; k++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q[q * head_dim + d] * K[k * head_dim + d];
            }
            scores[k] = dot * scale;
            if (scores[k] > max_score) max_score = scores[k];
        }

        // Softmax
        float sum = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            scores[k] = expf(scores[k] - max_score);
            sum += scores[k];
        }
        for (int k = 0; k < seq_len; k++) {
            scores[k] /= sum;
        }

        // Weighted sum of V
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                val += scores[k] * V[k * head_dim + d];
            }
            output[q * head_dim + d] = val;
        }
    }
    delete[] scores;
}

// ============================================================================
// Benchmark: Attention kernel with unified memory
// ============================================================================
double benchmark_attention_unified(int seq_len, int head_dim, int iterations) {
    size_t qkv_bytes = (size_t)seq_len * head_dim * sizeof(float);

    float *Q, *K, *V, *output;
    hipMallocManaged(&Q, qkv_bytes);
    hipMallocManaged(&K, qkv_bytes);
    hipMallocManaged(&V, qkv_bytes);
    hipMallocManaged(&output, qkv_bytes);

    // Initialize with pseudo-random values
    for (int i = 0; i < seq_len * head_dim; i++) {
        Q[i] = sinf((float)i * 0.01f) * 0.5f;
        K[i] = cosf((float)i * 0.013f) * 0.5f;
        V[i] = sinf((float)i * 0.017f + 1.0f) * 0.5f;
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(seq_len);
    dim3 block(1);  // One thread per query position

    // Prefetch to GPU
    int device;
    hipGetDevice(&device);
    hipMemPrefetchAsync(Q, qkv_bytes, device, 0);
    hipMemPrefetchAsync(K, qkv_bytes, device, 0);
    hipMemPrefetchAsync(V, qkv_bytes, device, 0);
    hipDeviceSynchronize();

    // Warmup
    hipLaunchKernelGGL(scaled_dot_product_attention,
                       grid, block, 0, 0,
                       Q, K, V, output, seq_len, head_dim, scale);
    hipDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(scaled_dot_product_attention,
                           grid, block, 0, 0,
                           Q, K, V, output, seq_len, head_dim, scale);
        hipDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count()
                / iterations;

    hipFree(Q); hipFree(K); hipFree(V); hipFree(output);
    return ms;
}

// ============================================================================
// Verification
// ============================================================================
bool verify_attention(int seq_len, int head_dim) {
    size_t qkv_bytes = (size_t)seq_len * head_dim * sizeof(float);

    float *Q, *K, *V, *gpu_out;
    hipMallocManaged(&Q, qkv_bytes);
    hipMallocManaged(&K, qkv_bytes);
    hipMallocManaged(&V, qkv_bytes);
    hipMallocManaged(&gpu_out, qkv_bytes);
    float* cpu_out = (float*)malloc(qkv_bytes);

    for (int i = 0; i < seq_len * head_dim; i++) {
        Q[i] = sinf((float)i * 0.01f) * 0.5f;
        K[i] = cosf((float)i * 0.013f) * 0.5f;
        V[i] = sinf((float)i * 0.017f + 1.0f) * 0.5f;
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    // GPU
    hipLaunchKernelGGL(scaled_dot_product_attention,
                       dim3(seq_len), dim3(1), 0, 0,
                       Q, K, V, gpu_out, seq_len, head_dim, scale);
    hipDeviceSynchronize();

    // CPU reference
    attention_cpu(Q, K, V, cpu_out, seq_len, head_dim, scale);

    // Compare
    float max_err = 0.0f;
    float avg_err = 0.0f;
    for (int i = 0; i < seq_len * head_dim; i++) {
        float err = fabsf(gpu_out[i] - cpu_out[i]);
        if (err > max_err) max_err = err;
        avg_err += err;
    }
    avg_err /= (float)(seq_len * head_dim);

    bool pass = max_err < 1e-4f;
    printf("  Attention verification (seq=%d, dim=%d): max=%.6e avg=%.6e %s\n",
           seq_len, head_dim, max_err, avg_err, pass ? "[PASS]" : "[FAIL]");

    hipFree(Q); hipFree(K); hipFree(V); hipFree(gpu_out);
    free(cpu_out);
    return pass;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("===========================================================\n");
    printf("  unified-ml: Custom Attention Kernel for gfx1150\n");
    printf("===========================================================\n\n");

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("GPU:           %s\n", props.name);
    printf("Architecture:  %s\n", props.gcnArchName);
    printf("Compute Units: %d\n\n", props.multiProcessorCount);

    // Verification
    printf("--- Correctness Check ---\n");
    verify_attention(32, 64);
    verify_attention(64, 64);
    verify_attention(128, 64);
    printf("\n");

    // Benchmarks
    printf("--- Attention Benchmark (Unified Memory) ---\n\n");
    printf("%-10s  %-10s  %-14s\n", "Seq Len", "Head Dim", "Time (ms)");
    printf("%-10s  %-10s  %-14s\n", "----------", "----------", "--------------");

    int seq_lens[]  = {64, 128, 256, 512};
    int head_dims[] = {64, 64, 64, 64};
    int count = 4;

    for (int i = 0; i < count; i++) {
        int iters = (seq_lens[i] <= 128) ? 50 : 10;
        double ms = benchmark_attention_unified(seq_lens[i], head_dims[i], iters);
        printf("%-10d  %-10d  %-14.3f\n", seq_lens[i], head_dims[i], ms);
    }

    printf("\n--- Key Design Choices ---\n");
    printf("1. KV-cache in unified memory: CPU manages cache eviction,\n");
    printf("   GPU reads K/V directly without copies\n");
    printf("2. One thread per query position (correctness-first PoC)\n");
    printf("3. Online softmax (numerically stable, single-pass)\n");
    printf("4. Production version would parallelize across head_dim\n");
    printf("   with warp-level reductions for higher throughput\n");

    return 0;
}
