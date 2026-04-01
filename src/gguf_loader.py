"""
gguf_loader.py — Zero-copy GGUF model loader for unified-ml.

Parses GGUF file headers, memory-maps tensor data, and dequantizes
common formats (F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) to float32 numpy arrays.

No external GGUF libraries — just struct + mmap + numpy.
"""

import struct
import mmap
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# GGUF magic number: ASCII "GGUF"
GGUF_MAGIC = 0x46554747  # little-endian "GGUF"

# GGUF metadata value types
GGUF_TYPE_UINT8    = 0
GGUF_TYPE_INT8     = 1
GGUF_TYPE_UINT16   = 2
GGUF_TYPE_INT16    = 3
GGUF_TYPE_UINT32   = 4
GGUF_TYPE_INT32    = 5
GGUF_TYPE_FLOAT32  = 6
GGUF_TYPE_BOOL     = 7
GGUF_TYPE_STRING   = 8
GGUF_TYPE_ARRAY    = 9
GGUF_TYPE_UINT64   = 10
GGUF_TYPE_INT64    = 11
GGUF_TYPE_FLOAT64  = 12

# GGML tensor types we support
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14

# Block sizes and byte sizes for quantized types
# Q4_0: 32 elements per block, 2 bytes scale + 16 bytes quants = 18 bytes/block
# Q8_0: 32 elements per block, 2 bytes scale + 32 bytes quants = 34 bytes/block
QUANT_INFO = {
    GGML_TYPE_F32:  {"name": "F32",  "block_size": 1,  "type_size": 4},
    GGML_TYPE_F16:  {"name": "F16",  "block_size": 1,  "type_size": 2},
    GGML_TYPE_Q4_0: {"name": "Q4_0", "block_size": 32, "type_size": 18},
    GGML_TYPE_Q8_0: {"name": "Q8_0", "block_size": 32, "type_size": 34},
    GGML_TYPE_Q4_K: {"name": "Q4_K", "block_size": 256, "type_size": 144},
    GGML_TYPE_Q5_K: {"name": "Q5_K", "block_size": 256, "type_size": 176},
    GGML_TYPE_Q6_K: {"name": "Q6_K", "block_size": 256, "type_size": 210},
}

# Extended type names for display
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
}


class GGUFLoader:
    """
    Memory-mapped GGUF model loader.

    Parses headers and tensor metadata, then provides zero-copy access
    to tensor data via mmap. Dequantizes Q4_0/Q8_0/Q4_K/Q5_K/Q6_K to float32 on demand.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

        self.file_size = self.path.stat().st_size
        self._fp = open(self.path, "rb")
        self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        self._offset = 0

        # Parsed data
        self.version = 0
        self.n_tensors = 0
        self.n_kv = 0
        self.metadata: Dict[str, Any] = {}
        self.tensor_infos: list = []
        self.data_offset = 0

        self._parse_header()
        self._parse_metadata()
        self._parse_tensor_infos()
        self._compute_data_offset()

    def close(self):
        """Release mmap and file handle."""
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fp:
            self._fp.close()
            self._fp = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    # ── Low-level readers ──────────────────────────────────────────

    def _read_bytes(self, n: int) -> bytes:
        data = self._mm[self._offset:self._offset + n]
        self._offset += n
        return data

    def _read_u8(self) -> int:
        return struct.unpack("<B", self._read_bytes(1))[0]

    def _read_i8(self) -> int:
        return struct.unpack("<b", self._read_bytes(1))[0]

    def _read_u16(self) -> int:
        return struct.unpack("<H", self._read_bytes(2))[0]

    def _read_i16(self) -> int:
        return struct.unpack("<h", self._read_bytes(2))[0]

    def _read_u32(self) -> int:
        return struct.unpack("<I", self._read_bytes(4))[0]

    def _read_i32(self) -> int:
        return struct.unpack("<i", self._read_bytes(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack("<Q", self._read_bytes(8))[0]

    def _read_i64(self) -> int:
        return struct.unpack("<q", self._read_bytes(8))[0]

    def _read_f32(self) -> float:
        return struct.unpack("<f", self._read_bytes(4))[0]

    def _read_f64(self) -> float:
        return struct.unpack("<d", self._read_bytes(8))[0]

    def _read_bool(self) -> bool:
        return self._read_u8() != 0

    def _read_string(self) -> str:
        length = self._read_u64()
        return self._read_bytes(length).decode("utf-8")

    def _read_value(self, vtype: int) -> Any:
        """Read a single metadata value by type tag."""
        readers = {
            GGUF_TYPE_UINT8:   self._read_u8,
            GGUF_TYPE_INT8:    self._read_i8,
            GGUF_TYPE_UINT16:  self._read_u16,
            GGUF_TYPE_INT16:   self._read_i16,
            GGUF_TYPE_UINT32:  self._read_u32,
            GGUF_TYPE_INT32:   self._read_i32,
            GGUF_TYPE_FLOAT32: self._read_f32,
            GGUF_TYPE_BOOL:    self._read_bool,
            GGUF_TYPE_STRING:  self._read_string,
            GGUF_TYPE_UINT64:  self._read_u64,
            GGUF_TYPE_INT64:   self._read_i64,
            GGUF_TYPE_FLOAT64: self._read_f64,
        }
        if vtype == GGUF_TYPE_ARRAY:
            return self._read_array()
        if vtype not in readers:
            raise ValueError(f"Unknown GGUF value type: {vtype}")
        return readers[vtype]()

    def _read_array(self) -> list:
        """Read a GGUF array value (type + length + elements)."""
        elem_type = self._read_u32()
        length = self._read_u64()
        return [self._read_value(elem_type) for _ in range(length)]

    # ── Header parsing ─────────────────────────────────────────────

    def _parse_header(self):
        """Parse GGUF file header: magic, version, tensor count, kv count."""
        magic = self._read_u32()
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Not a GGUF file (magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
            )
        self.version = self._read_u32()
        if self.version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {self.version} (need 2 or 3)")
        self.n_tensors = self._read_u64()
        self.n_kv = self._read_u64()

    def _parse_metadata(self):
        """Parse all key-value metadata pairs."""
        for _ in range(self.n_kv):
            key = self._read_string()
            vtype = self._read_u32()
            value = self._read_value(vtype)
            self.metadata[key] = value

    def _parse_tensor_infos(self):
        """Parse tensor info entries: name, dims, type, offset."""
        for _ in range(self.n_tensors):
            name = self._read_string()
            n_dims = self._read_u32()
            dims = [self._read_u64() for _ in range(n_dims)]
            dtype = self._read_u32()
            offset = self._read_u64()
            self.tensor_infos.append({
                "name": name,
                "dims": dims,
                "type": dtype,
                "type_name": GGML_TYPE_NAMES.get(dtype, f"type_{dtype}"),
                "offset": offset,
            })

    def _compute_data_offset(self):
        """Data section starts at the next 32-byte aligned boundary after headers."""
        self.data_offset = self._offset
        # GGUF aligns data to 32 bytes (GGUF_DEFAULT_ALIGNMENT)
        alignment = 32
        if self.data_offset % alignment != 0:
            self.data_offset += alignment - (self.data_offset % alignment)

    # ── Dequantization ─────────────────────────────────────────────

    @staticmethod
    def _dequant_q4_0(data: bytes, n_elements: int) -> np.ndarray:
        """
        Dequantize Q4_0 blocks to float32.
        Block layout: float16 scale (2 bytes) + 16 bytes of 4-bit quants = 18 bytes
        Each block holds 32 values (two 4-bit values per byte).
        """
        block_size = 32
        n_blocks = n_elements // block_size
        result = np.empty(n_elements, dtype=np.float32)

        buf = np.frombuffer(data, dtype=np.uint8)
        for i in range(n_blocks):
            block_start = i * 18
            # Scale is float16 stored in first 2 bytes
            scale = np.frombuffer(data[block_start:block_start + 2], dtype=np.float16)[0]
            scale = float(scale)
            # 16 bytes of packed 4-bit quants
            quants = buf[block_start + 2:block_start + 18]
            for j in range(16):
                q_lo = int(quants[j]) & 0x0F
                q_hi = (int(quants[j]) >> 4) & 0x0F
                result[i * block_size + j]      = scale * (q_lo - 8)
                result[i * block_size + j + 16]  = scale * (q_hi - 8)

        return result

    @staticmethod
    def _dequant_q4_0_fast(data: bytes, n_elements: int) -> np.ndarray:
        """Vectorized Q4_0 dequantization — much faster for large tensors."""
        block_size = 32
        bytes_per_block = 18
        n_blocks = n_elements // block_size

        raw = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
        raw = raw.reshape(n_blocks, bytes_per_block)

        # Extract scales (first 2 bytes of each block, interpret as float16)
        scales = raw[:, :2].copy().view(np.float16).astype(np.float32).flatten()

        # Extract quant bytes (bytes 2..17 of each block)
        qbytes = raw[:, 2:]  # shape: (n_blocks, 16)

        # Unpack low and high nibbles
        lo = (qbytes & 0x0F).astype(np.float32) - 8.0  # shape: (n_blocks, 16)
        hi = ((qbytes >> 4) & 0x0F).astype(np.float32) - 8.0

        # Interleave: first 16 are lo, next 16 are hi (per block)
        result = np.empty((n_blocks, block_size), dtype=np.float32)
        result[:, :16] = lo * scales[:, np.newaxis]
        result[:, 16:] = hi * scales[:, np.newaxis]

        return result.flatten()[:n_elements]

    @staticmethod
    def _dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
        """
        Vectorized Q8_0 dequantization.
        Block layout: float16 scale (2 bytes) + 32 int8 quants = 34 bytes
        Each block holds 32 values.
        """
        block_size = 32
        bytes_per_block = 34
        n_blocks = n_elements // block_size

        raw = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
        raw = raw.reshape(n_blocks, bytes_per_block)

        # Scales
        scales = raw[:, :2].copy().view(np.float16).astype(np.float32).flatten()

        # Quants as signed int8
        qbytes = raw[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)

        result = qbytes * scales[:, np.newaxis]
        return result.flatten()[:n_elements]

    # ── K-quant dequantization ────────────────────────────────────

    @staticmethod
    def _dequant_q4_k(data: bytes, n_elements: int) -> np.ndarray:
        """
        Vectorized Q4_K dequantization.

        Super-block layout (144 bytes, 256 values):
          - 2 bytes: f16 super-scale (d)
          - 2 bytes: f16 super-min (dmin)
          - 12 bytes: 6-bit packed scales/mins for 8 sub-blocks
          - 128 bytes: 4-bit quantized values (2 values per byte)

        Each super-block has 8 sub-blocks of 32 values.
        """
        block_size = 256
        bytes_per_block = 144
        n_blocks = n_elements // block_size

        raw = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
        raw = raw.reshape(n_blocks, bytes_per_block)

        # Super-scale d and dmin (f16)
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).flatten()
        dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()

        # 12 bytes of packed 6-bit scales/mins
        sc = raw[:, 4:16].astype(np.int32)

        sub_scales = np.empty((n_blocks, 8), dtype=np.float32)
        sub_mins = np.empty((n_blocks, 8), dtype=np.float32)

        # Sub-blocks 0-3: direct 6-bit values
        for j in range(4):
            sub_scales[:, j] = d * (sc[:, j] & 63)
            sub_mins[:, j] = dmin * (sc[:, j + 4] & 63)

        # Sub-blocks 4-7: combined from two sources
        for j in range(4, 8):
            sub_scales[:, j] = d * ((sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4))
            sub_mins[:, j] = dmin * ((sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4))

        # 128 bytes of 4-bit quantized values
        qbytes = raw[:, 16:]  # (n_blocks, 128)

        result = np.empty((n_blocks, block_size), dtype=np.float32)

        for j in range(8):
            qb = qbytes[:, j * 16:(j + 1) * 16]  # (n_blocks, 16)
            lo = (qb & 0xF).astype(np.float32)
            hi = ((qb >> 4) & 0xF).astype(np.float32)

            sd = sub_scales[:, j:j + 1]  # (n_blocks, 1)
            sm = sub_mins[:, j:j + 1]

            result[:, j * 32:j * 32 + 16] = lo * sd - sm
            result[:, j * 32 + 16:j * 32 + 32] = hi * sd - sm

        return result.flatten()[:n_elements]

    @staticmethod
    def _dequant_q5_k(data: bytes, n_elements: int) -> np.ndarray:
        """
        Vectorized Q5_K dequantization.

        Super-block layout (176 bytes, 256 values):
          - 2 bytes: f16 super-scale (d)
          - 2 bytes: f16 super-min (dmin)
          - 12 bytes: 6-bit packed scales/mins for 8 sub-blocks
          - 128 bytes: low 4-bit quantized values
          - 32 bytes: high bits (bit 4 of each quant value)

        5-bit quant = low 4 bits + 1 high bit.
        """
        block_size = 256
        bytes_per_block = 176
        n_blocks = n_elements // block_size

        raw = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
        raw = raw.reshape(n_blocks, bytes_per_block)

        # Super-scale d and dmin (f16)
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).flatten()
        dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()

        # 12 bytes of packed 6-bit scales/mins (same format as Q4_K)
        sc = raw[:, 4:16].astype(np.int32)

        sub_scales = np.empty((n_blocks, 8), dtype=np.float32)
        sub_mins = np.empty((n_blocks, 8), dtype=np.float32)

        for j in range(4):
            sub_scales[:, j] = d * (sc[:, j] & 63)
            sub_mins[:, j] = dmin * (sc[:, j + 4] & 63)

        for j in range(4, 8):
            sub_scales[:, j] = d * ((sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4))
            sub_mins[:, j] = dmin * ((sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4))

        # 128 bytes of low 4-bit quants
        ql = raw[:, 16:144]  # (n_blocks, 128)

        # 32 bytes of high bits — bit j of byte[j//8] is the high bit of element j
        qh = raw[:, 144:176]  # (n_blocks, 32)

        result = np.empty((n_blocks, block_size), dtype=np.float32)

        for j in range(8):
            qb = ql[:, j * 16:(j + 1) * 16]  # (n_blocks, 16)
            lo = (qb & 0xF).astype(np.int32)
            hi = ((qb >> 4) & 0xF).astype(np.int32)

            for k in range(16):
                elem_lo = j * 32 + k
                elem_hi = j * 32 + k + 16
                byte_lo = elem_lo // 8
                bit_lo = elem_lo % 8
                byte_hi = elem_hi // 8
                bit_hi = elem_hi % 8

                hb_lo = ((qh[:, byte_lo] >> bit_lo) & 1).astype(np.int32)
                hb_hi = ((qh[:, byte_hi] >> bit_hi) & 1).astype(np.int32)

                q_lo = lo[:, k] | (hb_lo << 4)
                q_hi = hi[:, k] | (hb_hi << 4)

                sd = sub_scales[:, j]
                sm = sub_mins[:, j]

                result[:, j * 32 + k] = sd * q_lo.astype(np.float32) - sm
                result[:, j * 32 + k + 16] = sd * q_hi.astype(np.float32) - sm

        return result.flatten()[:n_elements]

    @staticmethod
    def _dequant_q6_k(data: bytes, n_elements: int) -> np.ndarray:
        """
        Vectorized Q6_K dequantization.

        Super-block layout (210 bytes, 256 values):
          - 128 bytes: low 4-bit quantized values (ql)
          - 64 bytes: high 2-bit values (qh)
          - 16 bytes: int8 sub-block scales
          - 2 bytes: f16 super-scale (d)

        6-bit quant = low 4 bits from ql + high 2 bits from qh.
        Values: scale[sub] * d * (q - 32).
        """
        block_size = 256
        bytes_per_block = 210
        n_blocks = n_elements // block_size

        raw = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
        raw = raw.reshape(n_blocks, bytes_per_block)

        # Layout: ql(128) + qh(64) + scales(16) + d(2) = 210
        ql = raw[:, 0:128]
        qh = raw[:, 128:192]
        sc = raw[:, 192:208].view(np.int8).astype(np.float32)  # 16 int8 scales
        d = raw[:, 208:210].copy().view(np.float16).astype(np.float32).flatten()

        result = np.empty((n_blocks, block_size), dtype=np.float32)

        # Process in 2 halves of 128 elements, each reading 64 ql bytes + 32 qh bytes
        for n_off in range(2):
            n = n_off * 128
            ql_base = n_off * 64
            qh_base = n_off * 32

            ql_a = ql[:, ql_base:ql_base + 32]        # (n_blocks, 32)
            ql_b = ql[:, ql_base + 32:ql_base + 64]    # (n_blocks, 32)
            qh_c = qh[:, qh_base:qh_base + 32]         # (n_blocks, 32)

            # 4 groups of 32 values: combine low nibble/high nibble with 2-bit high
            q1 = (ql_a & 0xF).astype(np.int32) | (((qh_c >> 0) & 3).astype(np.int32) << 4)
            q2 = (ql_b & 0xF).astype(np.int32) | (((qh_c >> 2) & 3).astype(np.int32) << 4)
            q3 = ((ql_a >> 4) & 0xF).astype(np.int32) | (((qh_c >> 4) & 3).astype(np.int32) << 4)
            q4 = ((ql_b >> 4) & 0xF).astype(np.int32) | (((qh_c >> 6) & 3).astype(np.int32) << 4)

            is_base = n // 16  # 0 for first half, 8 for second half

            for q_vals, q_offset, sc_off in [
                (q1, 0, 0), (q2, 32, 2), (q3, 64, 4), (q4, 96, 6)
            ]:
                s_a = sc[:, is_base + sc_off] * d       # (n_blocks,)
                s_b = sc[:, is_base + sc_off + 1] * d   # (n_blocks,)

                vals_a = q_vals[:, :16].astype(np.float32) - 32.0
                vals_b = q_vals[:, 16:].astype(np.float32) - 32.0

                result[:, n + q_offset:n + q_offset + 16] = vals_a * s_a[:, np.newaxis]
                result[:, n + q_offset + 16:n + q_offset + 32] = vals_b * s_b[:, np.newaxis]

        return result.flatten()[:n_elements]

    # ── Public API ─────────────────────────────────────────────────

    def print_metadata(self):
        """Print model metadata in a readable format."""
        print(f"=== GGUF Model: {self.path.name} ===")
        print(f"Version:      {self.version}")
        print(f"Tensors:      {self.n_tensors}")
        print(f"KV pairs:     {self.n_kv}")
        print(f"File size:    {self.file_size / (1024**3):.2f} GB")
        print(f"Data offset:  0x{self.data_offset:X}")
        print()

        # Key metadata fields
        display_keys = [
            "general.architecture",
            "general.name",
            "general.file_type",
            "general.quantization_version",
            "{arch}.context_length",
            "{arch}.embedding_length",
            "{arch}.block_count",
            "{arch}.attention.head_count",
            "{arch}.attention.head_count_kv",
            "{arch}.vocab_size",
            "{arch}.feed_forward_length",
            "{arch}.rope.freq_base",
            "tokenizer.ggml.model",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        ]

        arch = self.metadata.get("general.architecture", "llama")

        print("--- Metadata ---")
        for key_template in display_keys:
            key = key_template.replace("{arch}", arch)
            if key in self.metadata:
                val = self.metadata[key]
                # Truncate long arrays/strings
                if isinstance(val, list) and len(val) > 5:
                    val = f"[{len(val)} items]"
                elif isinstance(val, str) and len(val) > 80:
                    val = val[:80] + "..."
                print(f"  {key}: {val}")

        # Print any extra general.* keys we missed
        for key, val in sorted(self.metadata.items()):
            if key.startswith("general.") and key not in [
                k.replace("{arch}", arch) for k in display_keys
            ]:
                if isinstance(val, list) and len(val) > 5:
                    val = f"[{len(val)} items]"
                elif isinstance(val, str) and len(val) > 80:
                    val = val[:80] + "..."
                print(f"  {key}: {val}")
        print()

    def print_tensor_summary(self, max_tensors: int = 20):
        """Print a summary of tensor names, shapes, and types."""
        print(f"--- Tensor Summary ({self.n_tensors} total) ---")
        for i, info in enumerate(self.tensor_infos[:max_tensors]):
            shape_str = "x".join(str(d) for d in info["dims"])
            n_elem = 1
            for d in info["dims"]:
                n_elem *= d
            qinfo = QUANT_INFO.get(info["type"])
            if qinfo:
                size_bytes = (n_elem // qinfo["block_size"]) * qinfo["type_size"]
            elif info["type"] in (10, 11, 12, 13, 14, 15):
                # K-quant types: approximate sizes per 256-element superblock
                k_type_sizes = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292}
                size_bytes = (n_elem // 256) * k_type_sizes.get(info["type"], 144)
            else:
                size_bytes = n_elem * 2  # fallback: assume ~2 bytes/element
            print(f"  [{i:3d}] {info['name']:50s}  {shape_str:>20s}  {info['type_name']:>6s}  {size_bytes/(1024**2):>8.2f} MB")
        if self.n_tensors > max_tensors:
            print(f"  ... and {self.n_tensors - max_tensors} more tensors")
        print()

    def get_tensor_data(self, name: str) -> np.ndarray:
        """
        Load and dequantize a single tensor by name.

        Returns a float32 numpy array. For F32/F16, this is a view or
        simple cast. For quantized types, dequantization is performed.
        """
        info = None
        for ti in self.tensor_infos:
            if ti["name"] == name:
                info = ti
                break
        if info is None:
            raise KeyError(f"Tensor not found: {name!r}")

        n_elements = 1
        for d in info["dims"]:
            n_elements *= d

        dtype = info["type"]
        qinfo = QUANT_INFO.get(dtype)
        if qinfo is None:
            raise ValueError(
                f"Unsupported tensor type: {info['type_name']} (id={dtype}) "
                f"for tensor {name!r}. Supported: F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K"
            )

        # Calculate byte size
        n_blocks = n_elements // qinfo["block_size"]
        byte_size = n_blocks * qinfo["type_size"]

        # Read from mmap
        offset = self.data_offset + info["offset"]
        raw = self._mm[offset:offset + byte_size]

        # Dequantize
        if dtype == GGML_TYPE_F32:
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        elif dtype == GGML_TYPE_F16:
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype == GGML_TYPE_Q4_0:
            arr = self._dequant_q4_0_fast(raw, n_elements)
        elif dtype == GGML_TYPE_Q8_0:
            arr = self._dequant_q8_0(raw, n_elements)
        elif dtype == GGML_TYPE_Q4_K:
            arr = self._dequant_q4_k(raw, n_elements)
        elif dtype == GGML_TYPE_Q5_K:
            arr = self._dequant_q5_k(raw, n_elements)
        elif dtype == GGML_TYPE_Q6_K:
            arr = self._dequant_q6_k(raw, n_elements)
        else:
            raise ValueError(f"No dequantizer for type {dtype}")

        # Reshape to original dimensions
        shape = tuple(reversed(info["dims"]))  # GGUF stores dims in reverse (row-major)
        return arr.reshape(shape)

    def load_all_tensors(self, supported_only: bool = True) -> Dict[str, np.ndarray]:
        """
        Load all tensors into a dict of {name: numpy_array}.

        If supported_only=True (default), skips tensors with unsupported
        quantization types. If False, raises on unsupported types.
        """
        tensors = {}
        skipped = 0
        for info in self.tensor_infos:
            if supported_only and info["type"] not in QUANT_INFO:
                skipped += 1
                continue
            tensors[info["name"]] = self.get_tensor_data(info["name"])

        if skipped:
            print(f"  [note] Skipped {skipped} tensors with unsupported quant types")
        return tensors


def load_gguf(path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Convenience function: load a GGUF file and return (metadata, tensors).

    Usage:
        metadata, tensors = load_gguf("model.gguf")
        print(metadata["general.architecture"])
        embed = tensors["token_embd.weight"]
    """
    loader = GGUFLoader(path)
    loader.print_metadata()
    loader.print_tensor_summary()
    tensors = loader.load_all_tensors()
    meta = dict(loader.metadata)
    loader.close()
    return meta, tensors


# ── CLI entrypoint ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gguf_loader.py <model.gguf> [--load-tensor <name>]")
        print()
        print("Parses GGUF header and prints model metadata + tensor summary.")
        print("With --load-tensor, dequantizes and prints stats for one tensor.")
        sys.exit(1)

    path = sys.argv[1]
    load_tensor_name = None
    if "--load-tensor" in sys.argv:
        idx = sys.argv.index("--load-tensor")
        if idx + 1 < len(sys.argv):
            load_tensor_name = sys.argv[idx + 1]

    with GGUFLoader(path) as loader:
        loader.print_metadata()
        loader.print_tensor_summary()

        if load_tensor_name:
            print(f"Loading tensor: {load_tensor_name}")
            arr = loader.get_tensor_data(load_tensor_name)
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            print(f"  Range: [{arr.min():.6f}, {arr.max():.6f}]")
            print(f"  Mean:  {arr.mean():.6f}")
            print(f"  Std:   {arr.std():.6f}")
            print(f"  Size:  {arr.nbytes / (1024**2):.2f} MB (float32)")
