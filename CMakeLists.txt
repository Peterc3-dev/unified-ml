cmake_minimum_required(VERSION 3.21)
project(unified-ml LANGUAGES CXX HIP)

# ============================================================================
# unified-ml: ML Inference Engine for AMD APU Unified Memory
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm / HIP configuration
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package(hip REQUIRED)

# Target architecture: gfx1150 (RDNA 3.5, Radeon 890M)
set(GPU_TARGETS "gfx1150" CACHE STRING "GPU architectures to compile for")

# ============================================================================
# Unified Memory Benchmark
# ============================================================================
add_executable(unified_benchmark src/unified_memory.hip)
target_compile_options(unified_benchmark PRIVATE
    -O3
    --offload-arch=${GPU_TARGETS}
)
target_link_libraries(unified_benchmark hip::device)

# ============================================================================
# Attention Kernel Benchmark
# ============================================================================
add_executable(attention_benchmark src/attention.hip)
target_compile_options(attention_benchmark PRIVATE
    -O3
    --offload-arch=${GPU_TARGETS}
)
target_link_libraries(attention_benchmark hip::device)

# ============================================================================
# MatMul Benchmark
# ============================================================================
add_executable(matmul_benchmark src/matmul.hip)
target_compile_options(matmul_benchmark PRIVATE
    -O3
    --offload-arch=${GPU_TARGETS}
)
target_link_libraries(matmul_benchmark hip::device)

# ============================================================================
# Install
# ============================================================================
install(TARGETS unified_benchmark attention_benchmark matmul_benchmark
        RUNTIME DESTINATION bin)
install(FILES include/unified_ml.h DESTINATION include)
