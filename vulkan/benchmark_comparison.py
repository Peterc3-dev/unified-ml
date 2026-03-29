"""HIP vs Vulkan vs NumPy comparison benchmark for unified-ml.

Runs matmul and memory bandwidth tests across backends and outputs results.
Requires: Kompute (in musicgen venv), numpy
"""
import json
import struct
import time
import sys
from pathlib import Path

import numpy as np

def benchmark_cpu_matmul(N=512, runs=10):
    times = []
    for i in range(runs + 2):
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        start = time.perf_counter()
        _ = a @ b
        elapsed = (time.perf_counter() - start) * 1000
        if i >= 2:
            times.append(elapsed)
    return {"avg_ms": round(sum(times)/len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2)}

def benchmark_gpu_matmul(N=512, runs=10):
    import kp
    mgr = kp.Manager()
    props = mgr.get_device_properties()
    spv_path = Path(__file__).parent / "matmul.spv"
    spirv = spv_path.read_bytes()
    size = N * N
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros(size, dtype=np.float32)
    t_a = mgr.tensor(a.tolist())
    t_b = mgr.tensor(b.tolist())
    t_c = mgr.tensor(c.tolist())
    params = [t_a, t_b, t_c]
    push_consts = [struct.unpack("f", struct.pack("I", N))[0]]
    times = []
    for i in range(runs + 2):
        start = time.perf_counter()
        seq = mgr.sequence()
        seq.record(kp.OpSyncDevice(params))
        algo = mgr.algorithm(params, spirv, [N // 16, N // 16, 1], [], push_consts)
        seq.record(kp.OpAlgoDispatch(algo))
        seq.record(kp.OpSyncLocal([t_c]))
        seq.eval()
        elapsed = (time.perf_counter() - start) * 1000
        if i >= 2:
            times.append(elapsed)
    # Verify
    t_c_data = np.array(t_c.data()).reshape(N, N)
    expected = a.reshape(N, N) @ b.reshape(N, N)
    max_err = float(np.max(np.abs(t_c_data - expected)))
    return {"avg_ms": round(sum(times)/len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "device": props["device_name"],
            "max_error": round(max_err, 6)}

def benchmark_gpu_bandwidth(N=1_000_000, runs=10):
    import kp
    mgr = kp.Manager()
    spv_path = Path(__file__).parent / "unified_memory.spv"
    spirv = spv_path.read_bytes()
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)
    t_a = mgr.tensor(a.tolist())
    t_b = mgr.tensor(b.tolist())
    t_c = mgr.tensor(c.tolist())
    params = [t_a, t_b, t_c]
    # push: num_elements as uint32, scale as float
    push_consts = [struct.unpack("f", struct.pack("I", N))[0], 1.0]
    times = []
    for i in range(runs + 2):
        start = time.perf_counter()
        seq = mgr.sequence()
        seq.record(kp.OpSyncDevice(params))
        algo = mgr.algorithm(params, spirv, [(N + 255) // 256, 1, 1], [], push_consts)
        seq.record(kp.OpAlgoDispatch(algo))
        seq.record(kp.OpSyncLocal([t_c]))
        seq.eval()
        elapsed = (time.perf_counter() - start) * 1000
        if i >= 2:
            times.append(elapsed)
    bytes_moved = N * 4 * 3  # 2 reads + 1 write
    avg_s = sum(times) / len(times) / 1000
    bandwidth_gbps = (bytes_moved / avg_s) / 1e9 if avg_s > 0 else 0
    return {"avg_ms": round(sum(times)/len(times), 2),
            "bandwidth_gbps": round(bandwidth_gbps, 2),
            "elements": N}

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"=== Unified-ML Benchmark Comparison (N={N}, runs={runs}) ===\n")
    
    print("CPU (NumPy) matmul...")
    cpu = benchmark_cpu_matmul(N, runs)
    print(f"  avg: {cpu['avg_ms']}ms  min: {cpu['min_ms']}ms  max: {cpu['max_ms']}ms\n")

    try:
        print("GPU (Vulkan/Kompute) matmul...")
        gpu = benchmark_gpu_matmul(N, runs)
        print(f"  avg: {gpu['avg_ms']}ms  min: {gpu['min_ms']}ms  max: {gpu['max_ms']}ms")
        print(f"  device: {gpu['device']}")
        print(f"  max_error: {gpu['max_error']}")
        speedup = cpu["avg_ms"] / gpu["avg_ms"] if gpu["avg_ms"] > 0 else 0
        print(f"  speedup vs CPU: {speedup:.1f}x\n")
    except Exception as e:
        gpu = {"status": f"error: {e}"}
        speedup = 0
        print(f"  ERROR: {e}\n")

    try:
        print("GPU memory bandwidth (1M elements)...")
        bw = benchmark_gpu_bandwidth(1_000_000, runs)
        print(f"  avg: {bw['avg_ms']}ms  bandwidth: {bw['bandwidth_gbps']} GB/s\n")
    except Exception as e:
        bw = {"status": f"error: {e}"}
        print(f"  ERROR: {e}\n")

    results = {
        "matmul_size": f"{N}x{N}",
        "runs": runs,
        "cpu_numpy": cpu,
        "gpu_vulkan": gpu,
        "gpu_bandwidth": bw,
        "speedup": round(speedup, 2),
    }
    
    out_path = "/tmp/vulkan_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
