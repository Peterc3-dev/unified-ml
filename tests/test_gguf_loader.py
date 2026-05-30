"""
Pure-logic unit tests for src/gguf_loader.py.

These tests exercise only the standalone numpy/struct/mmap logic:
GGUF header + metadata + tensor-info parsing, the 32-byte data-section
alignment rule, and the F32/F16/Q8_0/Q4_0 dequantizers. No GPU, ROCm,
torch, or model weights are involved — a minimal GGUF file is synthesized
in a tmp dir and parsed back.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable without installing the package.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from gguf_loader import (  # noqa: E402
    GGUF_MAGIC,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    GGML_TYPE_F32,
    GGUFLoader,
)


# ── Dequantizer tests (pure numpy math) ──────────────────────────────


def test_dequant_q8_0_roundtrip():
    """Q8_0: f16 scale + 32 int8 quants -> scale * quant."""
    scale = np.float16(0.5)
    quants = np.arange(-16, 16, dtype=np.int8)  # exactly one 32-value block
    data = scale.tobytes() + quants.tobytes()

    out = GGUFLoader._dequant_q8_0(data, 32)
    expected = quants.astype(np.float32) * float(scale)

    assert out.shape == (32,)
    assert np.allclose(out, expected)


def test_dequant_q4_0_fast_roundtrip():
    """Q4_0: f16 scale + 16 packed-nibble bytes; lo then hi, each minus 8."""
    scale = np.float16(0.25)
    # byte j packs lo=j and hi=j
    nibbles = bytes([(j) | (j << 4) for j in range(16)])
    data = scale.tobytes() + nibbles

    out = GGUFLoader._dequant_q4_0_fast(data, 32)

    lo = np.array([b & 0xF for b in nibbles], dtype=np.float32) - 8.0
    hi = np.array([(b >> 4) & 0xF for b in nibbles], dtype=np.float32) - 8.0
    expected = np.concatenate([lo, hi]) * float(scale)

    assert out.shape == (32,)
    assert np.allclose(out, expected)


def test_dequant_q4_0_scalar_matches_fast():
    """The reference scalar Q4_0 path must agree with the vectorized one."""
    scale = np.float16(0.125)
    nibbles = bytes([(j & 0xF) | (((15 - j) & 0xF) << 4) for j in range(16)])
    data = scale.tobytes() + nibbles

    fast = GGUFLoader._dequant_q4_0_fast(data, 32)
    scalar = GGUFLoader._dequant_q4_0(data, 32)

    assert np.allclose(fast, scalar)


def test_dequant_q8_0_multiple_blocks():
    """Two Q8_0 blocks with different scales dequantize independently."""
    s1, s2 = np.float16(1.0), np.float16(2.0)
    q1 = np.full(32, 3, dtype=np.int8)
    q2 = np.full(32, -4, dtype=np.int8)
    data = s1.tobytes() + q1.tobytes() + s2.tobytes() + q2.tobytes()

    out = GGUFLoader._dequant_q8_0(data, 64)

    assert np.allclose(out[:32], 3.0)
    assert np.allclose(out[32:], -8.0)


# ── Synthetic-GGUF parsing tests ─────────────────────────────────────


def _build_minimal_gguf(tmp_path: Path) -> Path:
    """
    Write a tiny but valid GGUF v3 file with one string metadata KV and one
    F32 tensor (4 elements). Returns the file path.
    """
    buf = bytearray()

    # Header: magic, version, n_tensors, n_kv
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 1)  # n_tensors
    buf += struct.pack("<Q", 1)  # n_kv

    # One KV: general.architecture = "llama"
    key = b"general.architecture"
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", GGUF_TYPE_STRING)
    val = b"llama"
    buf += struct.pack("<Q", len(val)) + val

    # One tensor info: name="t", 1 dim of 4, type F32, offset 0
    name = b"t"
    buf += struct.pack("<Q", len(name)) + name
    buf += struct.pack("<I", 1)  # n_dims
    buf += struct.pack("<Q", 4)  # dim 0
    buf += struct.pack("<I", GGML_TYPE_F32)
    buf += struct.pack("<Q", 0)  # offset within data section

    # Pad to 32-byte alignment, then write 4 float32 values.
    pad = (-len(buf)) % 32
    buf += b"\x00" * pad
    tensor_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf += tensor_vals.tobytes()

    path = tmp_path / "mini.gguf"
    path.write_bytes(bytes(buf))
    return path


def test_parse_minimal_gguf_header_and_metadata(tmp_path):
    path = _build_minimal_gguf(tmp_path)
    with GGUFLoader(str(path)) as loader:
        assert loader.version == 3
        assert loader.n_tensors == 1
        assert loader.n_kv == 1
        assert loader.metadata["general.architecture"] == "llama"
        assert loader.data_offset % 32 == 0


def test_parse_minimal_gguf_tensor_roundtrip(tmp_path):
    path = _build_minimal_gguf(tmp_path)
    with GGUFLoader(str(path)) as loader:
        assert len(loader.tensor_infos) == 1
        info = loader.tensor_infos[0]
        assert info["name"] == "t"
        assert info["type_name"] == "F32"

        arr = loader.get_tensor_data("t")
        assert arr.dtype == np.float32
        assert np.allclose(arr.flatten(), [1.0, 2.0, 3.0, 4.0])


def test_missing_tensor_raises(tmp_path):
    path = _build_minimal_gguf(tmp_path)
    with GGUFLoader(str(path)) as loader:
        with pytest.raises(KeyError):
            loader.get_tensor_data("does-not-exist")


def test_bad_magic_rejected(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(struct.pack("<I", 0xDEADBEEF) + b"\x00" * 32)
    with pytest.raises(ValueError):
        GGUFLoader(str(bad))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        GGUFLoader("/nonexistent/path/to/model.gguf")


def test_array_metadata_parsing(tmp_path):
    """A uint32 array metadata value parses into a Python list."""
    buf = bytearray()
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)  # n_tensors
    buf += struct.pack("<Q", 1)  # n_kv

    key = b"my.array"
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", 9)  # GGUF_TYPE_ARRAY
    buf += struct.pack("<I", GGUF_TYPE_UINT32)  # element type
    buf += struct.pack("<Q", 3)  # length
    buf += struct.pack("<I", 10) + struct.pack("<I", 20) + struct.pack("<I", 30)

    path = tmp_path / "arr.gguf"
    path.write_bytes(bytes(buf))
    with GGUFLoader(str(path)) as loader:
        assert loader.metadata["my.array"] == [10, 20, 30]
