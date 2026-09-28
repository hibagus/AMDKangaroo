# AMDKangaroo - CDNA 3 Optimization Strategy
## For MI300X (gfx942) and MI355X (gfx950)

**Document Version:** 1.0  
**Date:** September 28, 2026  
**Current Target:** RDNA 3 (gfx1100) at ~1300 Mk/s  
**Optimization Goal:** CDNA 3 (gfx942/gfx950) at 2000+ Mk/s  

---

## EXECUTIVE SUMMARY

This document provides a comprehensive optimization strategy to port AMDKangaroo from RDNA 3 (consumer/gaming GPUs) to CDNA 3 architecture (data-center/compute GPUs). The optimization focuses on exploiting CDNA 3's unique architectural features:

- **Wave64 native execution** (vs RDNA 3's Wave32)
- **256MB L2 cache** (42x larger than RDNA 3's 6MB)
- **192 CUs on Mi300X** (2x RDNA 3)
- **Better memory hierarchy** for scientific computing
- **Longer pipelines** with superior throughput

**Expected performance gain: 50-100%** (1300 Mk/s → 2000-2600 Mk/s)

---

## PART 1: ARCHITECTURAL ANALYSIS

### A. RDNA 3 Current Implementation (gfx1100)

**Configuration:**
```
BLOCK_SIZE = 256 threads/block
PNT_GROUP_CNT = 24 point groups per thread
Wave size = 32 (native for RDNA 3)
Occupancy = 100% with 1 wave/CU
L1 cache = 128KB per CU
L2 cache = 6MB shared
Memory bandwidth = 576 GB/s (7900 XTX)
Current performance = 1300 Mk/s
```

**Key insights:**
- Uses Wave32 efficiently for memory coalescing
- Limited L2 means working set must fit in L1
- High register pressure due to Wave32 (only 256KB per CU)
- BLOCK_SIZE=256 = 8 waves of 32 threads each

### B. CDNA 3 Architecture (gfx942/gfx950)

**Hardware Specifications:**

| Feature | MI300X (gfx942) | MI355X (gfx950) |
|---------|-----------------|-----------------|
| XCDs (Die Units) | 3 | 4+ |
| CUs per XCD | 64 | 64+ |
| Total CUs | 192 | 256+ |
| Wave size (native) | 64 | 64 |
| L1 Cache/CU | 128KB | 128KB |
| L2 Cache (total) | 256MB | 256MB |
| Registers/CU | 512KB (Wave64) | 512KB (Wave64) |
| Memory Bandwidth | 5.3 TB/s | 6.0 TB/s |
| Peak FP32 | 24.6 TF | 30+ TF |
| Peak FP64 | 24.6 TF | 30+ TF |
| Memory Type | HBM3e | HBM3e |
| Power | 750W | 900W+ |

**Key differences from RDNA 3:**
1. **Wave64 native**: Kernels written for Wave32 will have poor occupancy
2. **42x larger L2**: Can cache entire working sets (not just intermediate results)
3. **2x more CUs**: Better parallelism at higher levels
4. **Symmetric FP32/FP64**: ECC/precision features cost less
5. **Data-center focus**: Optimized for throughput over latency
6. **HBM3e memory**: Lower latency, higher bandwidth than GDDR6X

---

## PART 2: CURRENT CODEBASE ARCHITECTURE

### File Structure
```
AMDKangaroo/
├── defs.h                    # Kernel parameters & data structures
├── AMDGpuCore.hip            # GPU kernels (KernelA, B, C, Gen)
├── AMDGpuUtils.h             # Math primitives (conditional compilation)
├── AMDGpuUtils_AMD.h         # AMD portable C++ (no inline asm)
├── GpuKang.cpp               # GPU memory management, host-GPU sync
├── GpuKang.h                 # GPU class definition
├── AMDKangaroo.cpp           # GPU detection, device initialization
├── Ec.cpp/Ec.h               # Elliptic curve operations (CPU)
├── utils.cpp/utils.h         # Utilities
├── EcAsm.h                   # Assembly function declarations
├── secp256k1_asm_full.s      # x86-64 assembly (optional)
└── inverse256_skylake.s      # AVX2 modular inverse (optional)
```

### Memory Layout (SoA - Structure of Arrays)

**Current layout in global memory:**
```
Kangaroo data arranged for coalesced access:
[all_x0] [all_x1] [all_x2] [all_x3] [all_y0] [all_y1] [all_y2] [all_y3] [all_d0] [all_d1] [all_d2]
 ← 256b → ← 256b → ← 256b → ← 256b → ← 256b → ← 256b → ← 256b → ← 256b → ← 256b → ← 256b → ← 256b →

Stride = KangCnt (number of kangaroos)
KangCnt = BLOCK_SIZE × PNT_GROUP_CNT × BlockCnt
         = 256 × 24 × (CU_count / blocks_per_CU)
```

**KernelA dataflow:**
1. Load kangaroo state from global to registers (SoA pattern)
2. Copy jump tables to LDS (32KB)
3. Loop STEP_CNT=1000 times:
   - Load x coordinates for PNT_GROUP_CNT groups
   - Calculate inversions (batch modular inverse)
   - Perform EC point additions
   - Store back to global memory
   - Detect distinguished points (DP)
4. Write jump distances for KernelB

---

## PART 3: OPTIMIZATION PLAN

### Phase 1: Architecture Detection & Configuration (QUICK WIN - 1-2 commits)

**Objective:** Enable gfx942/gfx950 detection and add conditional compilation

**Changes needed:**

1. **Update defs.h** - Add CDNA 3 configuration:
```c
#elif defined(__HIP_PLATFORM_AMD__)
    #ifdef __gfx942__
        // MI300X - CDNA 3
        #define BLOCK_SIZE          512      // 8 waves of Wave64
        #define PNT_GROUP_CNT       12       // Reduced for better occupancy
        #define WAVE_SIZE           64
        #define CDNA3_OPT           1
    #elif defined(__gfx950__)
        // MI355X - CDNA 3
        #define BLOCK_SIZE          512      // 8 waves of Wave64
        #define PNT_GROUP_CNT       16       // Slightly higher for more parallelism
        #define WAVE_SIZE           64
        #define CDNA3_OPT           1
    #else
        // Default RDNA 3 (existing)
        #define BLOCK_SIZE          256
        #define PNT_GROUP_CNT       24
        #define WAVE_SIZE           32
    #endif
```

2. **Update Makefile** - Add CDNA 3 build targets:
```makefile
# Target selection
OFFLOAD_ARCH ?= gfx1100
# Usage: make OFFLOAD_ARCH=gfx942
# or:    make OFFLOAD_ARCH=gfx950

# Architecture-specific flags for CDNA 3
ifeq ($(OFFLOAD_ARCH),gfx942)
    HIPCCFLAGS += -D__gfx942__
endif
ifeq ($(OFFLOAD_ARCH),gfx950)
    HIPCCFLAGS += -D__gfx950__
endif

# Update linker target
$(TARGET): $(CPP_OBJECTS) $(HIP_OBJECTS) $(ASM_OBJECTS)
    $(HIPCC) --offload-arch=$(OFFLOAD_ARCH) -fgpu-rdc $(CCFLAGS) ...
```

3. **Update AMDKangaroo.cpp** - Improve GPU detection:
```cpp
// Detect CDNA 3 architecture
bool isCDNA3 = (deviceProp.major == 9 && deviceProp.minor == 4);  // gfx942
bool isCDNA35 = (deviceProp.major == 9 && deviceProp.minor == 5); // gfx950
GpuKangs[GpuCnt]->IsOldGpu = false;  // All CDNA 3 is modern
GpuKangs[GpuCnt]->IsCDNA3 = isCDNA3 || isCDNA35;
```

4. **Add documentation comment** to defs.h explaining Wave64 impact

**Commit message:**
```
Add CDNA 3 (gfx942/gfx950) architecture detection and configuration

- Add __gfx942__ and __gfx950__ conditional compilation branches
- Configure BLOCK_SIZE=512 (8 Wave64s) instead of Wave32
- Set PNT_GROUP_CNT=12 for gfx942, 16 for gfx950 (Wave64 reduces occupancy)
- Add Makefile target selection for OFFLOAD_ARCH
- Update GPU detection logic to identify CDNA 3 architecture

Note: Using Wave64 native execution reduces PNT_GROUP_CNT because Wave64 
uses more LDS per wave (doubling occupancy cost). This is compensated by 
CDNA 3's 2x CU count and superior memory hierarchy.
```

### Phase 2: Memory Layout Optimization for L2 Cache (HIGH IMPACT - 2-3 commits)

**Objective:** Leverage CDNA 3's massive L2 cache for caching entire working sets

**Current problem:**
- L2 cache (6MB on RDNA 3) = only ~1.5% of working set fits
- L2 needs refill on every kernel launch
- Register spilling due to limited L1 + L2

**Solution: L2-aware memory layout**

1. **Allocate persistent working set in L2 (GpuKang.cpp)**:
```cpp
// CDNA 3 specific: Use L2 cache persistence
if (IsCDNA3) {
    // Allocate jump tables to stay in L2
    hipMalloc(&Kparams.Jumps1, JMP_CNT * 96);
    hipMalloc(&Kparams.Jumps2, JMP_CNT * 96);
    hipMalloc(&Kparams.Jumps3, JMP_CNT * 96);
    
    // Hint: These should persist in L2
    // hipDeviceSetLimit(hipLimitPersistingL2CacheSize, 256MB);
    
    // Create virtual window for working set
    Kparams.L2WorkingSetSize = min(256MB, KangCnt * 96);
}
```

2. **Optimize SoA layout for L2 prefetching (AMDGpuCore.hip)**:
```hip
// CDNA 3: Better cache line utilization
// Current: 256-bit accesses (good for 64B cache lines)
// Optimize: Align jumps table to 128B for dual-issue loads
__align__(128) u64 jmp_table_cdna3[JMP_CNT * 8];

// Pre-fetch strategy for L2:
// Load all jmp_x, jmp_y at once (better L2 utilization)
for (int i = threadIdx.x; i < JMP_CNT; i += blockDim.x) {
    prefetch_l2(&jmp_table[8*i]);
}
```

3. **Add LDS pooling for Wave64 (AMDGpuCore.hip)**:
```hip
// Wave64 occupancy calculation:
// Each Wave64 uses 2x the LDS of Wave32
// Adjust LDS allocation to match Wave64 footprint

#ifdef CDNA3_OPT
    // CDNA 3: Wave64 uses more LDS
    // Reduce LDS usage by 30-40% through better packing
    Kparams.KernelA_LDS_Size = 40 * JMP_CNT + 8 * Kparams.BlockSize;  // 40KB instead of 64KB
    Kparams.KernelB_LDS_Size = 32 * JMP_CNT;  // 16KB instead of 64KB
#endif
```

4. **Create L2 cache working set descriptor**:
```cpp
struct L2WorkingSet {
    u64* jumps_combined;      // Jumps1/2/3 combined (96KB)
    u64* kangaroo_base;       // First batch of kangaroos
    u32  allocation_hint;     // For hipMemAdvise
};
```

**Commit messages:**
```
Commit 1: Optimize jump table caching for CDNA 3 L2 persistence
- Pre-allocate jump tables in persistent L2 region
- Align jump data to 128B cache line boundaries
- Add hipMemAdvise hints for L2 caching policy
- Expected improvement: 15-20% fewer L2 cache misses

Commit 2: Reduce LDS usage for Wave64 execution
- Pack jump list data more efficiently
- Use local arrays instead of LDS where possible
- Reduce KernelB LDS from 64KB to 32KB
- Reduce KernelA LDS from 64KB to 40KB

Commit 3: Implement L2-aware working set management
- Add L2WorkingSet descriptor for memory prefetching
- Implement non-temporal store hints for output
- Add memory advisor hints for CDNA 3 specific optimization
```

### Phase 3: Kernel Parameter Tuning (MEDIUM IMPACT - 1-2 commits)

**Objective:** Fine-tune BLOCK_SIZE and PNT_GROUP_CNT for Wave64

**Analysis:**

Current RDNA 3 (Wave32):
```
BLOCK_SIZE=256 = 8 waves of 32
Occupancy = 100% (full CU utilization)
Registers per wave = 8KB (less register pressure)
```

Proposed CDNA 3 (Wave64):
```
BLOCK_SIZE=512 = 8 waves of 64
Occupancy = same (8 waves fill CU), but better throughput
Registers per wave = 16KB (need lower PNT_GROUP_CNT)
```

**Tuning strategy:**

1. **Determine optimal BLOCK_SIZE** (test multiple values):
   - Option A: 512 (8×Wave64) - maximum occupancy
   - Option B: 256 (4×Wave64) - reduces register pressure
   - Option C: 384 (6×Wave64) - balance

2. **Calculate PNT_GROUP_CNT limits**:
```
Register budget per thread:
- Current usage: ~64 registers per thread
- Available on CDNA 3: 256 per thread (256KB per CU / 1024 threads per CU)
- With Wave64, effective limit: ~128 registers per thread (shared with waves)

Formula: PNT_GROUP_CNT = min(16, available_regs / regs_per_point_group)
- If regs_per_group = 8: max PNT_GROUP_CNT = 16
- If regs_per_group = 10: max PNT_GROUP_CNT = 12
```

3. **Proposed configurations to test**:

**Configuration A (Conservative):**
```
BLOCK_SIZE = 256 (4 waves of Wave64)
PNT_GROUP_CNT = 20
Expected occupancy: 75%
Expected improvement: 40-50% over RDNA 3
Rationale: Safer register usage, similar wave count to RDNA 3
```

**Configuration B (Aggressive):**
```
BLOCK_SIZE = 512 (8 waves of Wave64)
PNT_GROUP_CNT = 12
Expected occupancy: 100%
Expected improvement: 60-80% over RDNA 3
Rationale: Maximum CU utilization, matches CDNA 3 design
```

**Configuration C (MI355X variant):**
```
BLOCK_SIZE = 512 (8 waves of Wave64)
PNT_GROUP_CNT = 16
Expected occupancy: 100%
Expected improvement: 70-90% over RDNA 3
Rationale: More CUs available (256 vs 192), can afford more groups
```

4. **Dynamic configuration in GpuKang.cpp**:
```cpp
int AMDGpuKang::CalcKangCnt() {
    #ifdef __gfx942__
        Kparams.BlockSize = 512;
        Kparams.GroupCnt = 12;  // Conservative for gfx942
    #elif defined(__gfx950__)
        Kparams.BlockSize = 512;
        Kparams.GroupCnt = 16;  // Aggressive for gfx950
    #else
        // RDNA 3 existing config
        Kparams.BlockSize = 256;
        Kparams.GroupCnt = 24;
    #endif
    return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}
```

**Commit message:**
```
Tune kernel parameters for CDNA 3 Wave64 architecture

- Configure BLOCK_SIZE and PNT_GROUP_CNT per GPU:
  - gfx942 (MI300X): BLOCK_SIZE=512, PNT_GROUP_CNT=12
  - gfx950 (MI355X): BLOCK_SIZE=512, PNT_GROUP_CNT=16
- Account for Wave64 doubling register pressure per wave
- Maintain 100% occupancy while reducing register spilling
- Update CalcKangCnt() to use architecture-specific values

Performance impact: 60-80% improvement expected vs RDNA 3
```

### Phase 4: Loop Unrolling & Compiler Hints (MEDIUM IMPACT - 1 commit)

**Objective:** Maximize throughput through aggressive loop optimization

**Current state (defs.h):**
```c
#define STEP_CNT 1000  // Fixed loop iterations
```

**CDNA 3 optimization opportunities:**

1. **Increase STEP_CNT for CDNA 3** (longer pipelines benefit from larger working sets):
```c
#ifdef CDNA3_OPT
    #define STEP_CNT_CDNA3  1500  // 50% increase
#else
    #define STEP_CNT        1000
#endif
```

2. **Add pragma for loop unrolling** (Makefile):
```makefile
# Current flags:
HIPCCFLAGS += -mllvm -unroll-threshold=1000 \
              -mllvm -inline-threshold=10000

# Enhanced for CDNA 3:
HIPCCFLAGS += -mllvm -unroll-threshold=2000 \
              -mllvm -inline-threshold=20000 \
              -mllvm -unroll-allow-remainder=true
```

3. **Add __syncthreads() optimization hints** (AMDGpuCore.hip):
```hip
// CDNA 3: Synchronization is cheaper (Wave64 = 2x less busy-waiting)
// Add optimization pragma before main loop:
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams) {
    #pragma unroll(4)
    for (int step_ind = 0; step_ind < STEP_CNT; step_ind += 4) {
        // Process 4 steps with fewer sync points
        // ...
    }
}
```

4. **Optimize memory barriers**:
```hip
// CDNA 3: Can use __builtin_amdgcn_fence for finer control
// Current: __syncthreads() (heavy barrier)
// Proposed: Finer-grained memory ordering for Wave64
```

**Commit message:**
```
Add compiler optimizations for CDNA 3 throughput

- Increase STEP_CNT to 1500 for CDNA 3 (50% larger working set)
- Raise unroll-threshold from 1000 to 2000 for more aggressive unrolling
- Increase inline-threshold from 10000 to 20000 for better inlining
- Add pragma unroll hints for main loop iteration
- Enable unroll-allow-remainder for edge case handling

Expected impact: 5-10% throughput improvement through better pipeline utilization
```

### Phase 5: Wave64 Synchronization & Data Layout (ADVANCED - 2-3 commits)

**Objective:** Exploit Wave64 specific features for better convergence and data movement

**Key insight:** Wave64 has different execution characteristics:
- More threads per wave = better throughput
- More threads = larger synchronization domains
- Coalescing patterns change (64-element stride vs 32-element)

**Changes:**

1. **Update memory coalescing patterns** (AMDGpuCore.hip):
```hip
// Current (Wave32 optimized):
#define LOAD_VAL_256(dst, ptr, group) { \
    *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); \
    *((int4*)&(dst)[2]) = *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); \
}

// CDNA 3 (Wave64 optimized):
#ifdef CDNA3_OPT
    #define LOAD_VAL_256_W64(dst, ptr, group) { \
        int4 *p = (int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]; \
        *((int4*)&(dst)[0]) = p[0]; \
        *((int4*)&(dst)[2]) = p[2]; \
    }
#endif
```

2. **Implement Wave64-aware batch processing** (AMDGpuCore.hip):
```hip
// Wave64: Can process 64 point groups at once (vs 32 for Wave32)
// This improves work distribution:

#ifdef CDNA3_OPT
    // For each Wave64, load/process 64 elements
    u32 wave_id = threadIdx.x / 64;
    u32 lane_id = threadIdx.x % 64;
    
    // Better load balancing with larger waves
    for (int g = lane_id; g < PNT_GROUP_CNT; g += 64) {
        // Process point group g
    }
#endif
```

3. **Optimize __shfl operations for Wave64** (if used in InvModP):
```hip
// Wave64 shuffle patterns are different
// Current RDNA 3 optimizations for Wave32 need adjustment
#ifdef CDNA3_OPT
    // Wave64 shuffle has different cost profile
    // May want to reduce shuffle operations
    // Consider using LDS instead for larger synchronization domains
#endif
```

4. **Add sub-wave synchronization for CDNA 3**:
```cpp
// New macro for Wave64-aware barriers
#ifdef CDNA3_OPT
    #define SYNC_WAVE64() __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup")
#else
    #define SYNC_WAVE64() __syncthreads()
#endif
```

**Commit messages:**
```
Commit 1: Optimize memory access patterns for Wave64 (CDNA 3)
- Add LOAD_VAL_256_W64 macro for 64-element coalescing
- Adjust memory access stride patterns for larger waves
- Better cache line utilization (128B vs 64B)

Commit 2: Implement Wave64-aware thread scheduling
- Distribute work across 64-thread waves instead of 32-thread waves
- Improve load balancing within kernel
- Reduce thread divergence in conditional branches

Commit 3: Add CDNA 3 specific synchronization primitives
- Use __builtin_amdgcn_fence for finer control
- Optimize barrier placement for Wave64
- Reduce synchronization overhead
```

### Phase 6: Modular Arithmetic & Inverse Optimization (HIGH IMPACT - 2-3 commits)

**Objective:** Optimize the critical path (ModP operations) for CDNA 3

**Analysis:**
- InvModP is bottleneck (Fermat's little theorem: 254 squarings)
- MulModP/SqrModP called ~100K times per second
- These use portable C++ (no inline asm) for AMD compatibility

**Current AMD implementation (AMDGpuUtils_AMD.h):**
```cpp
#define mul_hi_64(res, a, b) { \
  __uint128_t temp = (__uint128_t)(a) * (__uint128_t)(b); \
  res = (u64)(temp >> 64); \
}
```

**CDNA 3 optimizations:**

1. **Use CDNA 3's native mul.wide** instruction via inline asm hints:
```cpp
// For CDNA 3 only - let compiler emit better code
#ifdef CDNA3_OPT
    // Hint: Use full 64×64→128 multiply
    __attribute__((always_inline))
    static inline u64 mul_hi_64_cdna3(u64 a, u64 b) {
        __uint128_t res = (__uint128_t)a * (__uint128_t)b;
        return res >> 64;
    }
#endif
```

2. **Unroll ModP multiply loops**:
```cpp
// Current: scalar loop
#define MulModP(res, a, b) { \
    u64 __carry = 0; \
    // ... loop for each 64-bit limb
}

// CDNA 3: Could potentially use 4-way VLIW to parallelize
// Or use pipelining hints
#ifdef CDNA3_OPT
    __attribute__((opencl_unroll_hint(4)))
    while (i < 4) { ... }
#endif
```

3. **Add vendor-specific reduce operations**:
```cpp
// CDNA 3: Can exploit wider ALUs
#ifdef __gfx942__
    // MI300X tuning: Focus on FP-heavy modular arithmetic
    #define MOD_REDUCE_UNROLL 4
#elif defined(__gfx950__)
    // MI355X tuning: Slightly more aggressive
    #define MOD_REDUCE_UNROLL 5
#endif
```

4. **Optimize Fermat's little theorem for CDNA 3**:
```cpp
// Current InvModP uses 254 SqrModP operations (exp 2^254 - 2)
// CDNA 3's larger register file allows more pipelining

#ifdef CDNA3_OPT
    // Parallelize independent sqr operations
    SqrModP(tmp1, inv);  // Can start while waiting for tmp1
    SqrModP(tmp2, inv);  // from previous iteration
    // Increases parallelism
#endif
```

**Commit messages:**
```
Commit 1: Optimize 64-bit multiply for CDNA 3
- Add mul_hi_64_cdna3 variant for better compiler output
- Use __uint128_t with optimization hints
- Expected: 5% improvement in ModP ops

Commit 2: Unroll ModP operation loops for pipelining
- Add __attribute__((opencl_unroll_hint)) for MulModP
- Increase parallelism in Fermat's little theorem computation
- Better utilize CDNA 3's longer pipelines
- Expected: 8-12% improvement in InvModP throughput

Commit 3: Add CDNA 3 specific multiply-reduce strategies
- Tailor inner loop optimization per GPU variant
- Use architecture-specific unroll factors
- Optimize for CDNA 3's register file size
```

### Phase 7: Advanced L2 Cache Management (OPTIONAL - 1-2 commits)

**Objective:** Fully exploit CDNA 3's 256MB L2 cache

**Note:** This is advanced and requires careful testing.

1. **Persistent L2 cache strategy**:
```cpp
// Use AMD's per-stream L2 policy (if supported in HIP)
#ifdef CDNA3_OPT
    // Hint: keep jump tables in L2
    hipStreamSetAttribute(..., hipStreamAttributeAccessPolicyWindow, 
                          window with jump_table pointers);
#endif
```

2. **Implement custom memory scheduler**:
```cpp
// Pre-load working set into L2 before kernel launch
void PreloadL2Cache(TKparams& params) {
    #ifdef CDNA3_OPT
        // Prefetch common access patterns
        for (u64* p = params.Jumps1; p < params.Jumps1 + JMP_CNT*12; p += 64) {
            __builtin_prefetch(p, 0, 3);  // Temporal, high locality
        }
    #endif
}
```

3. **Monitor L2 efficiency**:
```cpp
// Add performance counter collection
struct L2Stats {
    u64 accesses;
    u64 hits;
    u64 misses;
};
```

**Commit message:**
```
Add advanced L2 cache management for CDNA 3

- Implement per-stream L2 policy hints
- Add pre-loading strategy for working sets
- Insert performance monitoring for cache efficiency
- Expected impact: 10-15% improvement if L2 thrashing was an issue

Note: Requires careful profiling to validate - may not show gains on all workloads
```

---

## PART 4: IMPLEMENTATION ROADMAP

### Priority Order & Phasing

**Phase 1: Foundation (Week 1)**
- Duration: 2-3 days
- Commits: 3 small commits
- Risk: Minimal (configuration changes only)
- Expected impact: 0% (setup for later phases)
- Approach: Merge to feature branch for testing

**Phase 2: L2 Memory (Week 1-2)**
- Duration: 3-4 days
- Commits: 3 medium commits
- Risk: Medium (memory management changes)
- Expected impact: 15-20% improvement
- Approach: Test on actual hardware before merge

**Phase 3: Kernel Tuning (Week 2)**
- Duration: 2-3 days
- Commits: 1 commit with parameter sweeps
- Risk: Medium (performance tuning)
- Expected impact: 15-20% improvement
- Approach: Create test harness for parameter exploration

**Phase 4: Loop Optimization (Week 2-3)**
- Duration: 1-2 days
- Commits: 1 commit
- Risk: Low (compiler directives)
- Expected impact: 5-10% improvement
- Approach: Benchmark before/after

**Phase 5: Wave64 Features (Week 3)**
- Duration: 3-4 days
- Commits: 3 commits
- Risk: High (complex changes to kernel)
- Expected impact: 10-15% improvement
- Approach: Extensive testing, may need rollback

**Phase 6: Modular Arithmetic (Week 3-4)**
- Duration: 3-4 days
- Commits: 3 commits
- Risk: High (performance-critical code)
- Expected impact: 10-15% improvement
- Approach: Add test cases for correctness

**Phase 7: Cache Management (Week 4+)**
- Duration: 2-3 days
- Commits: 1-2 commits
- Risk: Medium (platform specific)
- Expected impact: Optional (10-15% if beneficial)
- Approach: Profiling-driven, may disable if not beneficial

### Total Timeline: 3-4 weeks
### Cumulative Expected Improvement: 60-100% (1300→2100-2600 Mk/s)

---

## PART 5: CODE ORGANIZATION & REFACTORING

### New File Structure (Post-optimization)

```
AMDKangaroo/
├── defs.h
│   ├── [NEW] Architecture detection macros
│   ├── [MODIFIED] Parameter definitions per GPU
│   └── [NEW] CDNA3-specific constants
├── config/
│   ├── [NEW] amd_rdna3.h      (Current config, BLOCK_SIZE=256, etc.)
│   ├── [NEW] amd_cdna3.h       (New config, BLOCK_SIZE=512, etc.)
│   ├── [NEW] compiler_flags.h  (Centralized compiler hints)
│   └── [NEW] memory_layout.h   (Memory optimization strategies)
├── AMDGpuCore.hip
│   ├── [MODIFIED] KernelA - add Wave64 optimizations
│   ├── [MODIFIED] KernelB - tune for new parameters
│   ├── [MODIFIED] KernelC - adapt sync points
│   └── [NEW] KernelA_wave64.hip (Alternative implementation if needed)
├── AMDGpuUtils.h              (No change - conditional compilation)
├── AMDGpuUtils_AMD.h
│   ├── [MODIFIED] Arithmetic ops - add CDNA3 variants
│   ├── [NEW] cdna3_arithmetic.h (CDNA3-specific math)
│   └── [NEW] L2_cache_hints.h   (Cache management macros)
├── GpuKang.cpp
│   ├── [MODIFIED] Memory allocation - L2 aware
│   ├── [NEW] CDNA3MemoryStrategy class
│   └── [NEW] Performance monitoring hooks
└── Makefile
    └── [MODIFIED] Multi-target build support
```

### Suggested Code Comments & Organization

**New comment headers for CDNA3 sections:**
```cpp
//=============================================================================
// CDNA 3 SPECIFIC OPTIMIZATION
// This code path is optimized for MI300X/MI355X (gfx942/gfx950)
// - Uses Wave64 natively for better throughput
// - Exploits 256MB L2 cache for working set caching
// - Adjusted for longer pipelines and higher bandwidth
//
// Do NOT mix CDNA 3 and RDNA 3 optimizations - test both independently
//=============================================================================
```

### Conditional Compilation Strategy

```cpp
// Tier 1: Architecture-level (required for correctness)
#ifdef __gfx942__
    // MI300X specific
#elif defined(__gfx950__)
    // MI355X specific
#else
    // RDNA 3 or other
#endif

// Tier 2: Feature-level (optional optimizations)
#ifdef CDNA3_OPT
    // All CDNA 3 GPUs (gfx942 and gfx950)
#endif

// Tier 3: Fallback for safety
#ifndef CDNA3_OPT
    #define SOME_PARAMETER RDNA3_DEFAULT_VALUE
#endif
```

---

## PART 6: PERFORMANCE TARGETS & BENCHMARKING

### Benchmark Methodology

**Setup:**
```bash
# Build for CDNA 3
make clean
make OFFLOAD_ARCH=gfx942

# Run test cases
./amdkangaroo -dp 16 -range 32 -start 100000000 \
  -pubkey 03a355aa5e2e09dd44bb46a4722e9336e9e3ee4ee4e7b7a0cf5785b283bf2ab579

./amdkangaroo -dp 16 -range 39 -start 8000000000 \
  -pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4
```

**Performance metrics:**
```
1. Mk/s (Mega-kangaroo-steps per second)
   - Primary metric
   - Measure as: Total jumps / (execution_time_seconds × 1,000,000)

2. Memory efficiency
   - L2 cache hit rate (if available)
   - Memory bandwidth utilization
   - Register utilization per CU

3. Power efficiency
   - Mk/s per Watt
   - Compare to RDNA 3 baseline

4. Correctness
   - Must produce same results as RDNA 3
   - Test on known puzzles
```

### Target Performance

| Phase | GPU | Config | Expected Mk/s | vs RDNA 3 |
|-------|-----|--------|---------------|----------|
| Baseline | RX7900XTX | RDNA3 | 1300 | 1.0x |
| Phase 1-3 | MI300X | gfx942 | 1950-2100 | 1.5-1.6x |
| Phase 1-6 | MI300X | gfx942 | 2400-2600 | 1.85-2.0x |
| Phase 1-3 | MI355X | gfx950 | 2100-2300 | 1.6-1.8x |
| Phase 1-6 | MI355X | gfx950 | 2800-3000 | 2.15-2.3x |

**Conservative estimate:** 50-80% improvement (1950-2350 Mk/s)
**Optimistic estimate:** 80-120% improvement (2400-2900 Mk/s)

---

## PART 7: QUICK REFERENCE - BUILD COMMANDS

### Build for different targets:
```bash
# RDNA 3 (existing, baseline)
make clean
make OFFLOAD_ARCH=gfx1100

# MI300X (CDNA 3)
make clean
make OFFLOAD_ARCH=gfx942

# MI355X (CDNA 3)
make clean
make OFFLOAD_ARCH=gfx950

# Auto-detect (if available)
make clean
make
```

### Runtime environment:
```bash
# Check which GPU is detected
rocminfo | grep gfx

# Monitor performance
rocm-smi --watch 1000

# Profile kernel execution
rocprof --hip-trace ./amdkangaroo -dp 16 -range 32 ...
```

---

## PART 8: RISK ASSESSMENT & CONTINGENCY

### Risk: Medium-High
- CDNA 3 is data-center focused, different optimization requirements
- Wave64 vs Wave32 is fundamental architectural difference
- No existing CDNA 3 GPU implementations to reference

### Contingency Plans:

**If performance is worse than expected:**
1. Reduce BLOCK_SIZE to 256 (fallback to 4 waves)
2. Increase PNT_GROUP_CNT back toward 24
3. Disable aggressive compiler flags
4. Use fallback to RDNA 3 config as emergency option

**If memory pressure is too high:**
1. Reduce JMP_CNT (fewer jump options)
2. Use unified memory instead of explicit GPU memory
3. Implement spilling strategy to HBM3e (slower but available)

**If Wave64 synchronization is problematic:**
1. Split into multiple smaller kernels
2. Use more LDS for temporary storage (256KB L2 per CU available)
3. Implement alternative synchronization patterns

---

## PART 9: SUCCESS CRITERIA

Optimization is successful if:

✅ **Functional correctness:**
- All test cases pass (Puzzle #33, #40, #85)
- Results match RDNA 3 baseline exactly
- No numerical discrepancies

✅ **Performance improvement:**
- Mi300X: ≥1900 Mk/s (50% improvement)
- Mi355X: ≥2100 Mk/s (60% improvement)
- Better performance than RDNA 3 on same power budget

✅ **Code quality:**
- Clear separation of CDNA 3 vs RDNA 3 code paths
- Well-commented sections explaining Wave64 changes
- Maintainable and extensible architecture

✅ **Production readiness:**
- Builds cleanly with no warnings
- Tested on actual MI300X/MI355X hardware
- Configuration documented clearly

---

## PART 10: APPENDIX - DETAILED TECHNICAL NOTES

### A. Wave64 vs Wave32 Execution

**Wave32 (RDNA 3):**
```
CU Layout:
├─ Wave 0: Threads 0-31
├─ Wave 1: Threads 32-63
├─ Wave 2: Threads 64-95
└─ Wave 3: Threads 96-127
(Up to 8-10 waves per CU for occupancy)

Coalescing pattern:
- Sequential threads access consecutive memory
- Stride = 32 threads × 8 bytes = 256 bytes per cache line
- Perfect for 64B cache lines with prefetching
```

**Wave64 (CDNA 3):**
```
CU Layout:
├─ Wave 0: Threads 0-63
└─ Wave 1: Threads 64-127
(Up to 4-5 waves per CU for occupancy)

Coalescing pattern:
- Sequential threads access consecutive memory
- Stride = 64 threads × 8 bytes = 512 bytes
- Can span multiple cache lines, potentially less efficient
- But: Higher throughput per wave compensates

Solution:
- Adjust coalescing patterns in LOAD_VAL/SAVE_VAL macros
- Use 2x load operations per thread (loading 512B across 2 cache lines)
```

### B. Occupancy Calculation

**RDNA 3 (BLOCK_SIZE=256, PNT_GROUP_CNT=24):**
```
Registers per thread: ~64
LDS per block: ~64KB
Max waves per CU: 10 (256 threads / 32 = 8 waves, but limited by resources)

Actual occupancy: 8 waves × 32 threads = 256 threads
Utilization: 256 / 2048 = 12.5% per CU (appears low, but is actually ~100% due to throughput)
```

**CDNA 3 (BLOCK_SIZE=512, PNT_GROUP_CNT=12):**
```
Registers per thread: ~64
LDS per block: ~40KB
Max waves per CU: 5 (512 threads / 64 = 8 waves, but limited by resources)

Actual occupancy: 8 waves × 64 threads = 512 threads
Utilization: 512 / 2048 = 25% per CU (Wave64 uses more register bandwidth)
But: Each wave has 2x throughput, so effective utilization is similar
```

### C. L2 Cache Sizing

**Current RDNA 3 usage:**
```
KangCnt = 294,912 kangaroos
Size = 294,912 × 96 bytes = ~28 MB per kangaroo working set
L2 = 6 MB

Working set / L2 = 4.7x (can only cache 21% of active data)
Result: High L2 miss rate, frequent DRAM accesses
```

**CDNA 3 opportunity:**
```
KangCnt = 294,912 × (2x MI300X CUs / 96 RDNA3 CUs) = ~600K kangaroos (roughly)
Size = 600,000 × 96 bytes = ~57 MB per working set (if proportional)
L2 = 256 MB

Working set / L2 = 0.22x (can cache 450% of working set!)
Result: L2 cache can hold entire working set + jumps + temporary buffers
Benefit: Nearly all accesses hit L2, eliminate DRAM stalls
```

### D. Memory Hierarchy on CDNA 3

```
Latency (approximate cycles):

L1 Cache (128KB per CU):
├─ Hit: 4-5 cycles
└─ Miss → L2

L2 Cache (256MB shared):
├─ Hit: 12-20 cycles
└─ Miss → HBM3e

HBM3e Memory:
├─ Latency: 300-500 cycles
└─ BUT: Bandwidth = 5.3 TB/s, masks latency with prefetching

Strategy:
- Keep high-reuse data in L1 (jump table x/y coordinates)
- Cache working set in L2 (all kangaroo states during STEP_CNT iterations)
- Stream output to HBM3e (distinguished points)
```

### E. Compiler Optimization Flags for CDNA 3

```makefile
# Recommended flags for gfx942/gfx950:

HIPCCFLAGS_CDNA3 := \
    -O3 \
    --offload-arch=gfx942 \
    -D__gfx942__ \
    -fgpu-rdc \
    -ffast-math \
    \
    -mllvm -amdgpu-early-inline-all=true \
    -mllvm -unroll-threshold=2000 \
    -mllvm -inline-threshold=20000 \
    -mllvm -unroll-allow-remainder=true \
    \
    -mllvm -amdgpu-dpp-combine-bitwise-or=true \
    -mllvm -amdgpu-enable-scratch-buffer=false \
    \
    $(HIPCCFLAGS)
```

---

## PART 11: CONCLUSION

This optimization strategy provides a phased, low-risk approach to leveraging CDNA 3's superior architecture for Pollard's Kangaroo ECDLP solving. By focusing on the key architectural differences (Wave64, L2 cache, CU count) and applying proven optimization techniques, we expect:

**Realistic improvement: 50-80% (1950-2350 Mk/s)**
**Optimistic improvement: 80-120% (2400-2900 Mk/s)**

The phased approach allows incremental validation and rollback capability at each stage, minimizing risk while maximizing learning and performance gains.

---

**Document Author:** Claude Haiku 4.5  
**Review Status:** Ready for Implementation  
**Last Updated:** September 28, 2026
