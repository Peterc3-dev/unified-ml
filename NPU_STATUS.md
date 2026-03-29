# NPU Proof of Life - GPD Pocket 4

**Date:** 2026-03-29
**Host:** GPD Pocket 4 (Tailscale 100.77.212.27)
**OS:** CachyOS (rolling), kernel 6.19.10-1-cachyos
**CPU:** AMD Ryzen AI 9 HX 370 (Strix Halo)

## Device Status: ALIVE

### Device Node
```
crw-rw-rw- 1 root render 261, 0 Mar 28 07:57 /dev/accel/accel0
```
- Permissions: 0666 (world read/write) -- no sudo needed for access
- User `raz` is in the `render` group (gid 988)

### Hardware Identity
```
PCI ID:     1022:17F0
PCI Slot:   0000:c6:00.1
Driver:     amdxdna
Device:     RyzenAI-npu4
Firmware:   1.0.0.63
Revision:   0x10
lspci:      Signal processing controller: AMD Strix/Krackan/Strix Halo Neural Processing Unit (rev 10)
```

### Kernel Driver
- `CONFIG_DRM_ACCEL_AMDXDNA=m` -- built as module in kernel config
- Module loaded and active:
  ```
  amdxdna               200704  1
  gpu_sched              73728  2 amdxdna,amdgpu
  ```
- The amdxdna module has 1 active reference (the NPU device)
- It shares gpu_sched with amdgpu (the Radeon 890M iGPU)

### Python Environment
- System Python: 3.14.3
- Python 3.12 available at `/usr/bin/python3.12`
- Python 3.11 available at `/usr/bin/python3.11`
- Venv created: `~/npu-venv` (python3.12)

### IREE Runtime
- **Installed:** iree-compiler 20241104.1068, iree-runtime 20241104.1068
- **Available HAL drivers:** cuda, hip, local-sync, local-task, vulkan
- **Note:** No dedicated XDNA/NPU HAL driver in this IREE release. The NPU requires either:
  - IREE built from source with XDNA support (iree-amd-aie project)
  - XRT (Xilinx Runtime) + mlir-aie toolchain
  - AMD's RyzenAI Software stack

### mlir_aie
- **Not available on PyPI** -- must be built from source via https://github.com/Xilinx/mlir-aie
- Expected: this is a source-only distribution

### XRT (Xilinx Runtime)
- **Not installed** (no /opt/xilinx, no xbutil/xrt-smi found in initial checks)
- Required for userspace NPU access beyond the raw /dev/accel device
- For Strix Halo (NPU4), the xdna-driver project provides the userspace stack

## Summary

| Component | Status |
|-----------|--------|
| NPU hardware | Present (PCI 1022:17F0) |
| Kernel driver (amdxdna) | Loaded, active, 1 ref |
| Device node (/dev/accel/accel0) | Exists, world-accessible |
| NPU firmware | v1.0.0.63 loaded |
| User permissions | OK (render group + 0666 perms) |
| Python 3.12 venv | Created at ~/npu-venv |
| IREE runtime | Installed (no XDNA HAL driver) |
| XRT userspace | NOT installed |
| mlir-aie | NOT installed (source-only) |

## Next Steps

1. **Install XRT/xdna-driver** -- The xdna-driver userspace stack is needed to talk to the NPU from applications. Repo: https://github.com/amd/xdna-driver -- requires building from source on Arch-based distros (no .deb/.rpm).

2. **Build IREE with XDNA support** -- The iree-amd-aie project (https://github.com/nod-ai/iree-amd-aie) adds an XDNA HAL driver to IREE, enabling compilation and execution of ML models on the NPU.

3. **Alternative: AMD RyzenAI Software** -- AMD provides a RyzenAI SDK with Vitis AI / ONNX Runtime EP for the NPU. This is the most turnkey path but targets Ubuntu officially.

4. **Quick validation** -- Once XRT is installed, `xbutil examine` should show the NPU device details and confirm the full userspace stack works.
