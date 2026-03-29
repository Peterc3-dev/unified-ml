# NPU Status — GPD Pocket 4 (Strix Halo)

**Last Updated:** 2026-03-29
**Host:** GPD Pocket 4 (Tailscale 100.77.212.27)
**OS:** CachyOS (rolling), kernel 6.19.10-1-cachyos
**CPU:** AMD Ryzen AI 9 HX 370 (Strix Halo)
**BIOS:** American Megatrends v2.10

## Current Status: WORKING (patched driver, no power management)

The NPU is fully operational using a patched out-of-tree amdxdna driver that
bypasses the non-functional SMU. FLM validates `ready: true`, inference runs.

### NPU Inference Benchmarks (Llama 3.2 1B via FLM)

| Metric | Value |
|--------|-------|
| Prefill speed | 40-46 tok/s |
| Decode speed | 14-24 tok/s |
| Haiku generation (14 tokens) | 1.68s wall clock |
| FLM validate | `ready: true`, 8 cols, FW 1.1.2.64 |

### Auto-load on Boot

The patched driver loads automatically via `npu-loader.service` (systemd).
It replaces the in-tree driver which fails on SMU init.

### The Fix

The SMU (System Management Unit) on Strix Halo is non-functional — it never
responds to commands regardless of firmware state. The fix: skip SMU init
entirely when PSP successfully loads firmware. The NPU runs without power
management (DPM/clock gating), using default clock speeds set by the BIOS.

Patch location: `~/builds/xrt-plugin/xdna-driver/drivers/accel/amdxdna/aie2_pci.c`

### The Original Error (for reference)

### The Error

```
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_smu_exec: smu cmd 4 failed, 0xff
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_smu_init: Access power failed, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hw_start: failed to init smu, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hw_resume: Start hardware failed, -22
```

- **SMU cmd 4** = `AIE_SMU_POWER_OFF` (0x4) — the driver sends POWER_OFF first to
  ensure a clean state, then POWER_ON. The POWER_OFF command itself fails.
- **0xff response** = SMU response register holds 0xFF (not `SMU_RESULT_OK` = 1).
  This means the SMU firmware is not running and the register holds a stale value.
- After PCI reset, the error changes to **timeout** (-110/ETIMEDOUT) because the
  reset clears the stale register value, and the SMU never writes a new response.

### Root Cause Analysis

**The NPU's SMU does not respond because the NPU firmware hasn't been loaded yet.**

The init sequence in `aie2_hw_start()` is:
1. `aie_smu_init()` — tries POWER_OFF + POWER_ON ← **FAILS HERE**
2. `aie_psp_start()` — loads firmware via PSP (Platform Security Processor)
3. `aie2_get_mgmt_chann_info()` — checks firmware is alive
4. Management firmware init, power management, etc.

On Strix Halo, the NPU's SMU is part of the NPU firmware (loaded by PSP), not a
standalone hardware component pre-initialized by the BIOS. The driver's init order
assumes the SMU is already running when the driver loads — this is wrong for Strix Halo.

**Evidence:**
- Direct BAR register reads confirm: SMU registers are accessible but the SMU
  never processes commands (CMD register holds the last written value, RESP stays 0)
- PSP is alive and responsive (`PSP_STATUS_READY` bit set in MP0_C2PMSG_123)
- PSP `RELEASE_TMR` command succeeds when sent directly via MMIO
- The BIOS does NOT pre-initialize the NPU's SMU for this hardware revision

### Hardware Identity

```
PCI ID:     1022:17F0 rev 0x10
PCI Slot:   0000:c6:00.1
IOMMU:      Group 25
NPU Type:   RyzenAI-npu4 (XDNA 2, aie2p architecture)
Capability: 50 TOPS INT8
BARs:
  BAR 0: [mem 0xdca00000-0xdcafffff] 1MB   — NPU registers + mailbox
  BAR 1: [mem 0xdcb00000-0xdcb01fff] 8K
  BAR 2: [mem 0x7810000000-0x781007ffff] 512K pref — SRAM
  BAR 4: [mem 0xdcb02000-0xdcb02fff] 4K    — PSP (Platform Security Processor)
  BAR 5: [mem 0xdcb03000-0xdcb03fff] 4K    — SMU (System Management Unit)
```

### SMU Register Dump (BAR 5)

```
SMU_CMD  (MP1_C2PMSG_0)  [0x900]: 0x00000004  ← POWER_OFF command (stale)
SMU_ARG  (MP1_C2PMSG_60) [0x9F0]: 0x00000000
SMU_RESP (MP1_C2PMSG_61) [0x9F4]: 0x00000000  ← No response (SMU not running)
SMU_INTR (APERTURE4_BASE)[0x000]: 0xFFFFFFFF  ← Interrupt reg (default/invalid)
```

### PSP Register State (BAR 4 + BAR 0)

```
PSP_STATUS (MP0_C2PMSG_123) [BAR4:0xAEC]: 0x80000000  ← READY bit set
PSP_INTR   (MP0_C2PMSG_73)  [BAR4:0xA24]: 0x00000001
SCRATCH3   (PSP_RESP)       [BAR0:0x10078]: 0xFFFF0007  ← PSP_ERROR_BAD_STATE
```

PSP is powered and responsive but returned BAD_STATE from a previous operation.

## Installed Software

### Kernel Driver

| Component | Version | Status |
|-----------|---------|--------|
| amdxdna.ko (in-tree) | kernel 6.19.10-1-cachyos | Loads, probe fails on SMU |
| amdxdna.ko (out-of-tree, patched) | from xdna-driver repo | Built, needs reboot to test |

### XRT Stack

| Package | Version |
|---------|---------|
| xrt | 2.21.75-5.1 |
| xrt-plugin-amdxdna | 2.21.75-2.1 |

### FastFlowLM (FLM)

| Component | Details |
|-----------|---------|
| Version | 0.9.37 |
| Binary | /usr/bin/flm |
| xclbin kernels | 38 model directories at /usr/share/flm/xclbins/ |
| Installed models | llama3.2:1b (Llama-3.2-1B) |
| Available models | 19 total (deepseek-r1:8b, gemma3, phi4-mini, qwen2.5, etc.) |

### NPU Firmware

| File | Size | Protocol |
|------|------|----------|
| npu.sbin.1.0.0.63.zst | 72 KB | Older |
| npu.sbin.1.1.2.64.zst | 125 KB | Newer |
| npu.sbin.zst (active) | → npu.sbin.1.1.2.64.zst | Symlink |
| npu_7.sbin.zst | → npu.sbin.1.1.2.64.zst | Protocol 7 symlink |
| npu.dev.sbin | NOT PRESENT | Development firmware |

Path: `/lib/firmware/amdnpu/17f0_10/`

### IREE

- Installed in ~/npu-venv (Python 3.12)
- Version: iree-compiler 20241104.1068, iree-runtime 20241104.1068
- HAL drivers: cuda, hip, local-sync, local-task, vulkan
- No XDNA HAL driver (requires iree-amd-aie built from source)

## IREE Benchmark Results

| Backend | Matrix Size | GFLOPS | Notes |
|---------|------------|--------|-------|
| CPU (local-task) | 1024x1024 | 27.4 | Multi-threaded |
| Vulkan (Radeon 890M) | 1024x1024 | 1,084.8 | RDNA 3 iGPU |
| NPU | — | — | Blocked (SMU failure) |

## What Was Tried

### Path A: Kernel 7.0+
- No kernel 7.0+ available in CachyOS repos (latest: 6.19.10)
- LTS kernel 6.18.20-1-cachyos-lts has amdxdna.ko (untested)

### Path B: PCI Device Reset
- `echo 1 > /sys/bus/pci/devices/0000:c6:00.1/reset` + module reload
- Result: Error changed from 0xff to **timeout** (worse — SMU completely unresponsive)

### Path C: Alternative Firmware
- Tried older firmware (npu.sbin.1.0.0.63.zst) — same timeout error
- Both firmware versions fail identically → not a firmware protocol issue

### Path D: PCI Remove + Rescan
- Removed NPU device from PCI bus and rescanned
- Result: Device re-enumerated correctly, same SMU timeout
- Side effect: GPU's SMU got stuck ("I'm not done with previous command") — caused display blackout

### Path E: Out-of-Tree Driver Build (PATCHED)
- Built from ~/builds/xrt-plugin/xdna-driver/ with init-order fix
- **Patch**: If SMU init fails in `aie2_hw_start()`, load firmware via PSP first, then retry SMU
- Build succeeded, patch triggered correctly ("SMU init failed, loading FW first")
- PSP timed out because register manipulation from Path D corrupted PSP state
- **Needs clean reboot to test properly**

### Path F: Direct PSP Communication
- Read PSP registers directly: PSP_STATUS_READY = true
- Sent PSP_RELEASE_TMR command via MMIO: succeeded (response = 0)
- Side effect: Released trusted memory region, crashed GPU display pipeline
- **Do NOT do this again** — direct PSP manipulation is dangerous on shared SoC

## The Fix (Pending Reboot Test)

Patched out-of-tree driver at:
```
~/builds/xrt-plugin/xdna-driver/drivers/accel/amdxdna/amdxdna.ko
```

Test script:
```bash
bash ~/builds/xrt-plugin/xdna-driver/test-patched-driver.sh
```

The patch modifies `aie2_hw_start()` in `aie2_pci.c`:
```c
// Original: SMU → PSP → FW alive
// Patched:  SMU → if fail → PSP → retry SMU → FW alive
ret = aie_smu_init(ndev->aie.smu_hdl);
if (ret) {
    XDNA_INFO(xdna, "SMU init failed (ret %d), loading FW first", ret);
    ret = aie_psp_start(ndev->aie.psp_hdl);
    if (ret) goto cleanup;
    ret = aie_smu_init(ndev->aie.smu_hdl);
    if (ret) goto cleanup;
} else {
    ret = aie_psp_start(ndev->aie.psp_hdl);
    if (ret) goto cleanup;
}
```

## If the Patch Doesn't Work

If SMU still doesn't respond after PSP loads firmware:

1. **Check BIOS settings**: NPU/IPU must be enabled, GPU mode set to "hybrid".
   Users have reported fixing identical SMU errors by changing BIOS GPU settings.

2. **Try npu.dev.sbin**: The development firmware from AMD's `drm-firmware` repo
   (`amd-ipu-staging` branch) may have different initialization behavior.

3. **Upstream kernel**: Kernel 7.0-rc2+ may have updated Strix Halo SMU support
   (per AMD Lemonade documentation).

4. **Skip SMU entirely**: If PSP loads firmware and FW_ALIVE is set, the driver
   could potentially operate without SMU power management (degraded mode).

## Available Kernel Modules

```
/lib/modules/6.19.10-1-cachyos/kernel/drivers/accel/amdxdna/amdxdna.ko.zst
/lib/modules/6.18.20-1-cachyos-lts/kernel/drivers/accel/amdxdna/amdxdna.ko.zst
```
