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
- **GPU:** AMD RDNA 3 (Radeon RX 7900 XTX tested, other RDNA 3 GPUs should work)
- **RAM:** 4+ GB recommended
- **Disk:** Minimal (< 1 GB)

### Software
- **OS:** Linux (tested on Ubuntu/Debian)
- **ROCm:** 6.0+ (tested with 6.4.3)
- **Compiler:** hipcc (comes with ROCm)
- **g++:** 11.4.0 or newer

## Installation

### 1. Install ROCm

```bash
# For Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocminfo | grep gfx1100
```

### 2. Build AMDKangaroo## Building

```bash
cd AMDKangaroo
make clean
make
```

The Makefile is configured for AMD RDNA 3 (gfx1100). If you have a different GPU, modify the `--offload-arch` flag in the Makefile.

### Build Configuration

The build uses three compilers:
- **hipcc** (ROCm) - GPU kernel compilation with aggressive optimizations
- **g++** - CPU code with `-O3 -march=native` for maximum performance  
- **as** (GNU assembler) - x86-64 assembly primitives (optional)

**Assembly Primitives:** Enabled by default for 10-20% performance boost. To disable, comment out `USE_ASM_PRIMITIVES := 1` in the Makefile.

**See:** `COMPILER_REQUIREMENTS.md` for detailed compiler flags and optimization settings.

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
GPU: -O3 --offload-arch=gfx1100 -ffast-math -munsafe-fp-atomics
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

1. **AMD-specific:** This port is optimized for AMD RDNA 3. Older AMD architectures may work but are untested.
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

**AMD Port Status:** ✅ Production Ready  
**Last Updated:** November 12, 2025  
**Tested On:** AMD Radeon RX 7900 XTX with ROCm 6.4.3
