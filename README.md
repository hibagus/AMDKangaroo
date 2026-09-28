# AMDKangaroo - AMD ROCm/HIP Port

**Fast GPU ECDLP solver using Kangaroo algorithm - Optimized for AMD RDNA 3 GPUs**

## Credits

- **Original Author:** RetiredCoder (RC) - https://github.com/RetiredC
- **AMD Port:** Sirius437 (2025)
- **License:** GPLv3

This is a port of the AMDKangaroo CUDA application to AMD ROCm/HIP, specifically optimized for AMD RDNA 3 architecture (Radeon RX 7900 XTX and similar GPUs).

## What is AMDKangaroo?

AMDKangaroo is a fast GPU implementation of the Pollard's Kangaroo algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP). It's designed for educational purposes and demonstrates state-of-the-art GPU acceleration techniques.

## AMD Port Features

### ✅ Fully Functional
- Complete CUDA to HIP conversion
- All NVIDIA PTX assembly converted to portable C++ (AMD best practice)
- Optimized for AMD RDNA 3 architecture (gfx1100)
- 96 Compute Units fully utilized
- Aggressive compiler optimizations
- Verified correct results on multiple test cases

### 🚀 Performance
- **1300 Mk/s** on AMD Radeon RX 7900 XTX 
- Optimized kernel parameters for RDNA 3 architecture
- Structure-of-Arrays (SoA) memory layout for better coalescing
- x86-64 assembly primitives for host-side operations
- Wave32 native execution

### 🔧 Technical Highlights
- **x86-64 Assembly Optimizations:** AVX2-optimized modular inverse (Bernstein & Yang)
- **GPU Kernel Tuning:** BLOCK_SIZE=256, PNT_GROUP_CNT=16, JMP_CNT=512
- **Memory Layout:** SoA for coalesced GPU access
- **Compiler flags:** Aggressive inlining, loop unrolling, vectorization
- **Architecture-specific:** gfx1100 (RDNA 3) optimizations

## System Requirements

### Hardware
- **GPU:** AMD Radeon RX 7900 XTX (`gfx1100`), AMD Instinct MI300X (`gfx942`),
  or AMD Instinct MI355X (`gfx950`)
- **RAM:** 4+ GB recommended
- **Disk:** Minimal (< 1 GB)

### Software
- **OS:** Linux (tested on Ubuntu/Debian)
- **ROCm:** 6.0+ for RDNA 3; current CDNA development uses ROCm 7.2
- **Compiler:** hipcc (comes with ROCm)
- **g++:** 11.4.0 or newer
- **Boost headers:** Multiprecision is required only to build the independent
  host reference used by the GPU field-arithmetic test

## Installation

### 1. Install ROCm

```bash
# For Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocminfo | grep -E 'gfx1100|gfx942|gfx950'
```

### 2. Build AMDKangaroo

```bash
cd AMDKangaroo
make clean
make
```

The default build targets `gfx950` and retains the historical `amdkangaroo`
executable name. Explicit architecture targets create binaries that can coexist:

```sh
make gfx942
make gfx950
make gfx1100

# Build both CDNA targets.
make all-cdna
```

- `amdkangaroo-gfx942` targets MI300X.
- `amdkangaroo-gfx950` targets MI355X.
- `amdkangaroo-gfx1100` preserves the Radeon RX 7900 XTX build.

Every architecture uses a separate object directory, so switching targets
cannot reuse an incompatible GPU object. At startup, each binary reads HIP's
`gcnArchName` and accepts only the architecture for which it was compiled.
Runtime profiles distinguish RDNA WGP reporting from CDNA CU reporting, and
host launches use the same geometry constants as the compiled kernels.
Fatal HIP errors identify the device and failed operation, stop the affected
worker after its first error, and terminate the solve cleanly if no GPU workers
remain.
AMD field multiplication and squaring fold carries beyond bit 255 and return a
canonical value below the secp256k1 prime. The square accumulator also reserves
an explicit carry limb so valid high-valued inputs cannot corrupt local state.
KernelB keeps an independent `MD_LEN`-entry distance history for every
block, point group, and thread lane. Its loads and stores now use
`threadIdx.x` for the innermost lane index, preventing all threads in a block
from accidentally sharing one lane's history.

Broader CDNA correctness and performance validation are still in progress.
Current status is tracked in the
[CDNA implementation plan](docs/CDNA_IMPLEMENTATION_PLAN.md) for current status.

### Build Configuration

The build uses three compilers:
- **hipcc** (ROCm) - GPU kernel compilation for the selected `GPU_ARCH`
- **g++** - CPU code with `-O3 -march=native` for maximum performance  
- **as** (GNU assembler) - x86-64 assembly primitives (optional)

**Assembly primitives:** Enabled by default. To disable them, comment out
`USE_ASM_PRIMITIVES := 1` in the Makefile.

### Deterministic GPU Field-Arithmetic Test

The test calls the same device functions used by the solver and compares 1,409
fixed cases with an independent Boost.Multiprecision reference. It covers
modular negate, add, subtract, multiply, square, and inverse operations across
zero, one, `p - 1`, carry- and borrow-heavy boundaries, and fixed-seed random
inputs.

```sh
# Compile test binaries for both CDNA targets without running them.
make field-tests-cdna

# Run on MI355X GPU 0. Choose another visible device with FIELD_TEST_GPU.
make test-field-gfx950 FIELD_TEST_GPU=0

# Run this only on an MI300X host.
make test-field-gfx942 FIELD_TEST_GPU=0
```

Each test binary contains one GPU architecture and rejects an incompatible
device before launching. The current MI355X result is 1,409/1,409 passing;
MI300X code generation passes, but runtime validation still requires MI300X
hardware.

### Deterministic GPU Kernel-State Test

The state test compiles the production `KernelGen`, `KernelA`, and `KernelB`
with one jump per launch. It compares point coordinates and signed distances
with the CPU secp256k1 implementation after 1, 2, 10, and 100 jumps. Its samples
span point-group, lane, and wave boundaries, and it separately checks that each
lane receives its own first-step loop-history value.

```sh
# Compile the state-test binaries for both CDNA targets.
make state-tests-cdna

# Run on MI355X GPU 0.
make test-state-gfx950 STATE_TEST_GPU=0

# Run this only on an MI300X host.
make test-state-gfx942 STATE_TEST_GPU=0
```

The MI355X test passes all four checkpoints for 16 sampled kangaroos. The
`gfx942` binary compiles successfully, but runtime validation still requires
MI300X hardware.

### Known-Puzzle Regression

The end-to-end regression runs the normal 1,000-step production kernels against
the documented 32-bit puzzle, requires the exact expected private key, and uses
an isolated temporary directory so `RESULTS.TXT` does not alter the checkout.

```sh
make test-known-puzzle-gfx950 PUZZLE_TEST_GPU=0

# Run this only on an MI300X host.
make test-known-puzzle-gfx942 PUZZLE_TEST_GPU=0
```

The default timeout is 60 seconds and can be changed with
`PUZZLE_TEST_TIMEOUT=<seconds>`. This regression passes on MI355X; the MI300X
run remains pending hardware access.

### Reproducible Kernel Benchmark

Benchmark mode creates a deterministic workload for one explicitly selected
GPU, discards warm-up iterations, and reports the median of the timed samples.
It times KernelA, KernelB, KernelC, device-buffer clearing, result transfers,
and the complete host-observed iteration separately.

```sh
./amdkangaroo-gfx950 -benchmark -gpu 0 -range 78 -dp 16 -bench-warmup 3 -bench-iterations 9 -bench-seed 0x5EED950942 -bench-output /tmp/mi355x-baseline.csv
```

The CSV appends one row per timed sample plus a median row. It records the
fixed seed and workload, GPU UUID and architecture, ROCm driver/runtime
versions, launch profile, nominal clocks, and optional pre/post temperature,
power, and clock samples from `rocm-smi`. Missing telemetry does not invalidate
the HIP event timings. Pre/post samples provide environmental context; they are
not in-kernel measurements and can show an idle clock immediately after a run.

`raw_gjumps_s` uses KernelA time because KernelA performs the configured
`STEP_CNT` jumps. `effective_gjumps_s` uses the complete iteration time and
therefore includes KernelB, KernelC, clearing, transfers, and host dispatch
overhead. Compare runs only when the architecture, seed, range, DP setting,
warm-up count, timed count, and workload constants are identical.

Benchmark-specific options are:

- `-benchmark`: enter deterministic benchmark mode.
- `-bench-output FILE`: append samples to this required CSV file.
- `-bench-warmup N`: warm-up iterations, default 3.
- `-bench-iterations N`: timed iterations, default 9.
- `-bench-seed N`: decimal or `0x`-prefixed seed, default `0x5EED950942`.

Use the linked-code-object report to capture register, spill, scratch, LDS,
wave-size, and code-size baselines for every production kernel:

```sh
make kernel-resources-gfx950

# This compiles and reports gfx942 resources without running on MI300X.
make kernel-resources-gfx942
```

The report is CSV on standard output, so it can be redirected to an experiment
artifact. Resource counts alone are not a performance result; retain the
benchmark CSV and correctness-test results with every optimization comparison.

The current ROCm 7.2 CDNA builds let the AMD backend choose inlining and
unrolling instead of forcing very large LLVM thresholds. For KernelA on both
`gfx942` and `gfx950`, that change reduced the final linked resources as
follows:

- VGPRs: 512 to 208
- VGPR spills: 240 to 0
- Scratch bytes per thread: 964 to 0
- Code size: 63,780 to 49,752 bytes

On MI355X, the identical 3-warm-up/9-timed seeded benchmark changed median
effective throughput from 4.35583 to 4.36201 GJumps/s (about 0.14%). That is
within normal run-to-run noise, so it is not claimed as a speedup. The change
removes scratch traffic and creates register headroom for later occupancy and
launch-geometry tuning without regressing the measured baseline.

## Usage

### Basic Command
```bash
./amdkangaroo -dp <DP_BITS> -range <BIT_RANGE> -start <START_VALUE> -pubkey <PUBLIC_KEY>
```

### Parameters
- **-dp**: Distinguished Point bits (14-60, recommended: 16)
- **-range**: Bit range of private key (32-170)
- **-start**: Starting value for search
- **-pubkey**: Public key to solve (compressed format, 33 bytes hex)

### Example: Puzzle #33 (32-bit)
```bash
./amdkangaroo -dp 16 -range 32 -start 100000000 \
  -pubkey 03a355aa5e2e09dd44bb46a4722e9336e9e3ee4ee4e7b7a0cf5785b283bf2ab579
```

**Expected:** Solves in < 1 second  
**Result:** `PRIVATE KEY: 00000000000000000000000000000000000000000000000000000001A96CA8D8`

### Example: Puzzle #40 (39-bit)
```bash
./amdkangaroo -dp 16 -range 39 -start 8000000000 \
  -pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4
```

**Expected:** Solves in < 1 second  
**Result:** `PRIVATE KEY: 000000000000000000000000000000000000000000000000000000E9AE4933D6`

### Example: Puzzle #85 (84-bit)
```bash
./amdkangaroo -dp 16 -range 84 -start 1000000000000000000000 \
  -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a
```

**Expected:** Takes significant time (2^42 operations estimated)

### Example: Puzzle #135 (134-bit)

./amdkangaroo -dp 60 -range 134 -start 4000000000000000000000000000000000 -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16

## GPU Configuration

Your AMD 7900 XTX will be detected with:
```
GPU 0: Radeon RX 7900 XTX, 23.98 GB, 96 CUs, cap 11.0, PCI 18, L2 size: 6144 KB
GPU 0: allocated 905 MB, 294912 kangaroos. OldGpuMode: No
```

- **Architecture:** RDNA 3 (gfx1100)
- **Mode:** Modern GPU (OldGpuMode: No)
- **Kangaroos:** 294,912
- **Memory:** ~905 MB allocated
- **Block Size:** 256 threads
- **Point Groups:** 24

## Output

When a key is found:
1. **Console:** Displays the private key
2. **File:** Writes to `RESULTS.TXT`

Example output:
```
Stopping work ...
Point solved, K: 795.495 (with DP and GPU overheads)

PRIVATE KEY: 000000000000000000000000000000000000000000000000000000E9AE4933D6
```

## Conversion Details

### What Was Changed

1. **All CUDA API calls → HIP equivalents**
   - `cudaMalloc` → `hipMalloc`
   - `cudaMemcpy` → `hipMemcpy`
   - `cudaGetDeviceProperties` → `hipGetDeviceProperties`
   - etc.

2. **PTX Assembly → Portable C++**
   - 520+ lines of NVIDIA PTX inline assembly
   - Converted to portable C++ using 128-bit arithmetic
   - Explicit carry tracking with local variables
   - All device functions ported (MulModP, SqrModP, InvModP, AddModP, SubModP, etc.)

3. **Architecture Detection**
   - Added AMD RDNA 3 detection
   - Proper CU count reporting (96 CUs, not 48 WGPs)
   - Modern GPU path for RDNA 3

4. **Compiler Optimizations**
   - Aggressive inlining and loop unrolling
   - Fast math operations
   - Architecture-specific tuning
   - Vectorization enabled

5. **x86-64 Assembly Optimizations (Optional)**
   - Optimized AddModP, SubModP, MulModP primitives
   - AVX2-optimized modular inverse from Bernstein & Yang
   - 10-20% estimated performance improvement
   - Can be enabled/disabled at compile time

### Files Modified/Created
- `Makefile` - ROCm/HIP build system
- `AMDKangaroo.cpp` - HIP API calls, AMD GPU detection
- `GpuKang.cpp` - HIP memory management
- `AMDGpuCore.hip` - GPU kernel file (from .cu)
- `defs.h` - AMD architecture detection
- `AMDGpuUtils.h` - Conditional compilation
- `AMDGpuUtils_AMD.h` - **NEW:** Complete portable C++ implementation (875 lines)

## Troubleshooting

### GPU Not Detected
```bash
# Check ROCm installation
rocminfo | grep gfx1100

# Check HIP devices
hipconfig --platform
```

### Build Fails
```bash
# Verify ROCm path
ls /opt/rocm/bin/hipcc

# Check environment
echo $PATH | grep rocm
```

### Kernel Fails
- Check that you're using the modern GPU path
- Look for "OldGpuMode: No" in output
- Should allocate ~294K kangaroos, not 1.5M

## Performance

### Verified Performance
- **Puzzle #33:** < 1 second ✅
- **Puzzle #40:** < 1 second ✅
- **Puzzle #85:** Running (estimated 2^42 operations)

### Compiler Optimizations Applied
```makefile
CPU: -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize
GPU: -O3 --offload-arch=$(GPU_ARCH) -ffast-math -munsafe-fp-atomics
     -mllvm -amdgpu-early-inline-all=true
     -mllvm -unroll-threshold=1000 -mllvm -inline-threshold=10000
```

## Advanced Options

### Generate Tames
```bash
./amdkangaroo -dp 16 -range 76 -tames tames76.dat -max 10
```

### Use Pre-generated Tames
```bash
./amdkangaroo -dp 16 -range 76 -start <VALUE> -pubkey <KEY> -tames tames76.dat
```

### Limit Operations
```bash
./amdkangaroo -dp 16 -range 84 -start <VALUE> -pubkey <KEY> -max 5.5
```

## Technical Documentation

For detailed technical information about the port, see:
- [MI300X and MI355X implementation plan](docs/CDNA_IMPLEMENTATION_PLAN.md)
- Original project: https://github.com/RetiredC
- AMD ROCm documentation: https://rocm.docs.amd.com/
- HIP programming guide: https://rocm.docs.amd.com/projects/HIP/

## Known Limitations

1. **CDNA bring-up:** `gfx942` and `gfx950` build targets are available, but
   end-to-end correctness and performance validation are still in progress.
   MI300X runtime validation requires access to an MI300X system.
2. **Linux only:** ROCm primarily supports Linux. Windows support via WSL2 is experimental.
3. **Single GPU:** Multi-GPU support exists but is untested on AMD.

## Contributing

Contributions are welcome! Areas for improvement:
- Further performance optimizations
- Further compiler optimizations
- Multi-GPU testing and optimization 

## License

This software is licensed under GPLv3. See LICENSE.TXT for details.

## Acknowledgments

- **RetiredCoder (RC)** for the original CUDA implementation
- **AMD** for ROCm and HIP
- **GPUOpen** for optimization resources and documentation
- **Daniel J. Bernstein and Bo-Yin Yang** for the fast constant-time modular inverse algorithm
  - Paper: "Fast constant-time gcd computation and modular inversion" (CHES 2019)
  - Implementation: https://gcd.cr.yp.to/index.html
  - Used in the AVX2-optimized inverse256_skylake assembly code

## Disclaimer

This software is for educational and research purposes only. The author is not responsible for any misuse of this software.

---

- **RDNA 3 port status:** Existing `gfx1100` implementation
- **CDNA port status:** Bring-up in progress
- **Last updated:** September 28, 2026
- **RDNA 3 tested on:** AMD Radeon RX 7900 XTX with ROCm 6.4.3
- **CDNA smoke-tested on:** AMD Instinct MI355X (`gfx950`) with ROCm 7.2
