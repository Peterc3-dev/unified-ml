"""
unified-ml: Python interface for the AMD APU unified memory inference engine.

Wraps the compiled HIP binaries via subprocess for benchmarking,
and provides ctypes hooks for the shared library (when built).
"""

import subprocess
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List

# Project root (assumes standard layout)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"


class UnifiedMLError(Exception):
    """Raised when a unified-ml operation fails."""
    pass


class DeviceInfo:
    """AMD APU device information."""
    def __init__(self, name: str, arch: str, compute_units: int,
                 max_clock_mhz: int, memory_mb: int, is_apu: bool):
        self.name = name
        self.arch = arch
        self.compute_units = compute_units
        self.max_clock_mhz = max_clock_mhz
        self.memory_mb = memory_mb
        self.is_apu = is_apu

    def __repr__(self):
        return (f"DeviceInfo(name={self.name!r}, arch={self.arch!r}, "
                f"CUs={self.compute_units}, clock={self.max_clock_mhz}MHz, "
                f"mem={self.memory_mb}MB, APU={self.is_apu})")


class UnifiedML:
    """
    Python interface to unified-ml HIP benchmarks.
    
    Usage:
        engine = UnifiedML()
        info = engine.device_info()
        results = engine.run_memory_benchmark()
    """

    def __init__(self, build_dir: Optional[Path] = None):
        self.build_dir = Path(build_dir) if build_dir else BUILD_DIR
        self._check_binaries()

    def _check_binaries(self):
        """Verify that the benchmark binaries exist."""
        self.binaries = {}
        for name in ["unified_benchmark", "attention_benchmark", "matmul_benchmark"]:
            path = self.build_dir / name
            if path.exists():
                self.binaries[name] = path
        if not self.binaries:
            raise UnifiedMLError(
                f"No benchmark binaries found in {self.build_dir}. "
                f"Build first with: hipcc -O3 --offload-arch=gfx1150 ..."
            )

    def _run_binary(self, name: str, timeout: int = 120) -> str:
        """Run a benchmark binary and return its stdout."""
        if name not in self.binaries:
            raise UnifiedMLError(f"Binary {name!r} not found. Available: {list(self.binaries.keys())}")
        
        result = subprocess.run(
            [str(self.binaries[name])],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.build_dir),
        )
        if result.returncode != 0:
            raise UnifiedMLError(
                f"{name} failed (rc={result.returncode}):\n{result.stderr}"
            )
        return result.stdout

    def run_memory_benchmark(self) -> str:
        """Run the unified memory vs standard HIP benchmark."""
        return self._run_binary("unified_benchmark")

    def run_attention_benchmark(self) -> str:
        """Run the custom attention kernel benchmark."""
        return self._run_binary("attention_benchmark")

    def run_matmul_benchmark(self) -> str:
        """Run the tiled matmul benchmark."""
        return self._run_binary("matmul_benchmark")

    def run_all_benchmarks(self) -> Dict[str, str]:
        """Run all available benchmarks and return results."""
        results = {}
        for name in self.binaries:
            short = name.replace("_benchmark", "")
            print(f"Running {short} benchmark...")
            start = time.time()
            results[short] = self._run_binary(name)
            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s")
        return results

    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Run all benchmarks and save results to a file."""
        if output_path is None:
            output_path = self.build_dir / "benchmark_results.txt"

        results = self.run_all_benchmarks()

        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("  unified-ml Benchmark Results\n")
            f.write(f"  Generated: {time.strftime(%Y-%m-%d %H:%M:%S)}\n")
            f.write("=" * 60 + "\n\n")

            for name, output in results.items():
                f.write(output)
                f.write("\n" + "-" * 60 + "\n\n")

        print(f"Results saved to {output_path}")
        return output_path


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="unified-ml benchmark runner")
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR,
                        help="Path to build directory containing binaries")
    parser.add_argument("--benchmark", choices=["memory", "attention", "matmul", "all"],
                        default="all", help="Which benchmark to run")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save results to file")
    args = parser.parse_args()

    engine = UnifiedML(build_dir=args.build_dir)

    if args.benchmark == "all":
        if args.save:
            engine.save_results(args.save)
        else:
            results = engine.run_all_benchmarks()
            for name, output in results.items():
                print(output)
    else:
        method = {
            "memory": engine.run_memory_benchmark,
            "attention": engine.run_attention_benchmark,
            "matmul": engine.run_matmul_benchmark,
        }[args.benchmark]
        print(method())


if __name__ == "__main__":
    main()
