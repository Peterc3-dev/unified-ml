"""
Pure-logic unit tests for python/unified_ml.py.

unified_ml.py only depends on the standard library (subprocess/pathlib/
argparse) — no GPU or HIP binaries are executed here. These tests cover
the binary-discovery bookkeeping, error handling, and DeviceInfo repr,
using fake executables in a tmp dir.
"""

from pathlib import Path
import sys

PKG = Path(__file__).resolve().parent.parent / "python"
sys.path.insert(0, str(PKG))

import pytest  # noqa: E402

from unified_ml import DeviceInfo, UnifiedML, UnifiedMLError  # noqa: E402


def test_device_info_repr_roundtrip():
    info = DeviceInfo(
        name="Radeon 890M",
        arch="gfx1150",
        compute_units=16,
        max_clock_mhz=2900,
        memory_mb=8192,
        is_apu=True,
    )
    r = repr(info)
    assert "Radeon 890M" in r
    assert "gfx1150" in r
    assert "CUs=16" in r
    assert "APU=True" in r


def test_no_binaries_raises(tmp_path):
    with pytest.raises(UnifiedMLError):
        UnifiedML(build_dir=tmp_path)


def test_binary_discovery_finds_existing(tmp_path):
    # Create two of the three known benchmark binaries.
    (tmp_path / "matmul_benchmark").write_text("#!/bin/sh\necho hi\n")
    (tmp_path / "unified_benchmark").write_text("#!/bin/sh\necho hi\n")

    engine = UnifiedML(build_dir=tmp_path)
    assert set(engine.binaries) == {"matmul_benchmark", "unified_benchmark"}
    assert "attention_benchmark" not in engine.binaries


def test_run_unknown_binary_raises(tmp_path):
    (tmp_path / "matmul_benchmark").write_text("#!/bin/sh\necho hi\n")
    engine = UnifiedML(build_dir=tmp_path)
    with pytest.raises(UnifiedMLError):
        engine._run_binary("attention_benchmark")
