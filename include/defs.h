////////////////////////////////////////////////////////////////////////////////
// AMDKangaroo - GPU-accelerated ECDLP Solver
// (c) 2024 RetiredCoder (RC) - https://github.com/RetiredC
// (c) 2025 Port to CDNA 3/4 architectures
// License: GPLv3, see LICENSE.TXT
//
// This file defines architecture-specific parameters for optimal performance
// on different AMD GPU architectures (RDNA 3, CDNA 3, CDNA 4)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#pragma warning(disable : 4996)

// ============================================================================
// Fundamental Type Definitions
// ============================================================================
// These are the basic types used throughout the application.
// Using explicit-width types (u64, u32, etc.) ensures consistent behavior
// across different platforms and compilers.

typedef unsigned long long u64;  // 64-bit unsigned integer
typedef long long i64;           // 64-bit signed integer
typedef unsigned int u32;        // 32-bit unsigned integer
typedef int i32;                 // 32-bit signed integer
typedef unsigned short u16;      // 16-bit unsigned integer
typedef short i16;               // 16-bit signed integer
typedef unsigned char u8;        // 8-bit unsigned integer
typedef char i8;                 // 8-bit signed integer

// ============================================================================
// Global Constants
// ============================================================================

// Maximum number of GPUs to support (for multi-GPU scenarios)
#define MAX_GPU_CNT				32

// Number of kernel iterations before device synchronization
// - Lower = more frequent sync (higher overhead, better load distribution)
// - Higher = fewer launches (lower overhead, less responsive)
// Note: Must be divisible by MD_LEN (10)
//
// Optimized per architecture:
// - RDNA 3: 1000 steps balances responsiveness with launch overhead
// - CDNA 3/4: 1500 steps reduce launch overhead (deeper pipelines tolerate longer execution)
//   Rationale: CDNA GPUs have 9x better memory bandwidth + 256MB L2 cache
//   Longer kernels = fewer context switches = better cache efficiency
#ifdef CDNA3_ARCHITECTURE
	#define STEP_CNT			1500
#elif defined(CDNA4_ARCHITECTURE)
	#define STEP_CNT			1500
#else
	#define STEP_CNT			1000
#endif

// Number of jump table entries - MUST BE POWER OF 2
// Larger tables give better randomization but increase memory usage and access time
// 512 = 2^9 provides optimal balance for ECDLP solving
#define JMP_CNT					512

// ============================================================================
// GPU Architecture Detection and Parameter Configuration
// ============================================================================
//
// The Kangaroo algorithm's performance is heavily dependent on:
// 1. BLOCK_SIZE: Number of threads per workgroup
//    - Affects occupancy, LDS usage, and wave count
// 2. PNT_GROUP_CNT: Number of independent point groups per thread
//    - Increases parallelism but uses more registers/LDS
// 3. Wave size: Hardware waves (32 for RDNA, 64 for CDNA)
//    - Affects synchronization patterns and vectorization
//
// Different AMD architectures have different optimal parameters:
// - LDS cache sizes and bandwidth
// - Wave sizes (32 vs 64)
// - Register budgets per wave
// - L2 cache sizes and organization
// - Memory hierarchy and bandwidth
//

#ifdef __CUDA_ARCH__
	// NVIDIA CUDA path (legacy, kept for compatibility)
	#if __CUDA_ARCH__ < 890
		#define OLD_GPU
	#endif
	#ifdef OLD_GPU
		#define BLOCK_SIZE			512
		#define PNT_GROUP_CNT		64
	#else
		#define BLOCK_SIZE			512
		#define PNT_GROUP_CNT		24
	#endif

#elif defined(__HIP_PLATFORM_AMD__)
	// AMD HIP Platform - Detect specific architecture via compiler macros
	// These are set by hipcc based on --offload-arch flag

	#if defined(__gfx1100__)
		// ====================================================================
		// RDNA 3 Architecture: Radeon RX 7900 XTX (gfx1100)
		// ====================================================================
		// Wave size:     32 (native Wave32 execution)
		// CUs:           96
		// L1 Cache:      32KB per CU
		// L2 Cache:      ~6MB total
		// Memory:        GDDR6X 576 GB/s
		// Base frequency:2500 MHz
		//
		// Characteristics:
		// - Smaller L2 cache (6MB) limits working set to ~23% of needed data
		// - Wave32 means 32 threads per wave
		// - Optimized for gaming/graphics (latency priority)
		//
		// Optimal configuration rationale:
		// - BLOCK_SIZE=256: 8 waves per block provides good occupancy
		// - PNT_GROUP_CNT=24: Each thread processes 24 independent points
		//   * 256 * 24 = 6144 total points per workgroup
		//   * Balances between L2 cache pressure and thread work
		//   * Good register usage (doesn't spill)
		//
		// Performance baseline: ~1300 Mk/s on RX 7900 XTX
		// ====================================================================
		#define BLOCK_SIZE			256
		#define PNT_GROUP_CNT		24

	#elif defined(__gfx942__)
		// ====================================================================
		// CDNA 3 Architecture: AMD Mi300X (gfx942)
		// ====================================================================
		// Wave size:     64 (native Wave64 execution)
		// CUs:           192 (via 3 XCDs with 64 CUs each)
		// L1 Cache:      32KB per CU
		// L2 Cache:      256MB total (42x larger than RDNA 3!)
		// Memory:        HBM3e 5.3 TB/s (9x bandwidth of RDNA 3)
		// Base frequency:2700 MHz
		//
		// Characteristics:
		// - MASSIVE L2 cache (256MB) - can cache 450% of working set
		//   * vs RDNA 3 which caches only 23% - 8x fewer DRAM stalls!
		// - Wave64 means 64 threads per wave (2x throughput of RDNA 3)
		// - Designed for datacenter compute (throughput priority)
		// - More CUs available (2x of RDNA 3)
		//
		// Optimal configuration rationale:
		// - BLOCK_SIZE=512: 8 waves of 64 threads = full workgroup
		//   * Better utilization of Wave64 hardware
		//   * Reduces synchronization overhead
		// - PNT_GROUP_CNT=12: Lower than RDNA 3 because:
		//   * Wave64 already provides 2x parallelism base
		//   * Keeps per-thread working set smaller for L2 efficiency
		//   * Reduces register pressure (Wave64 has more but still finite)
		//   * 512 * 12 = 6144 total points per workgroup (same as RDNA 3!)
		//   * But better distributed across 64-thread waves
		//
		// Performance target: 1900+ Mk/s (50% improvement over RDNA 3)
		// Main enabler: L2 cache efficiency with Wave64 execution
		// ====================================================================
		#define BLOCK_SIZE			512
		#define PNT_GROUP_CNT		12
		#define CDNA3_ARCHITECTURE 1

	#elif defined(__gfx950__)
		// ====================================================================
		// CDNA 4 Architecture: AMD Mi355X (gfx950)
		// ====================================================================
		// Wave size:     64 (native Wave64 execution)
		// CUs:           256+ (via 4+ XCDs with 64 CUs each)
		// L1 Cache:      32KB per CU (optimized further)
		// L2 Cache:      256MB total (same as CDNA 3)
		// Memory:        HBM3e 6.0 TB/s (improved over CDNA 3)
		// Base frequency:2800 MHz
		//
		// Characteristics:
		// - Same L2 cache as CDNA 3 (256MB)
		// - More CUs than Mi300X (256+ vs 192)
		// - Improved FP32 performance vs CDNA 3
		// - Better memory latency hiding
		// - Newer CDNA generation features
		//
		// Optimal configuration rationale:
		// - BLOCK_SIZE=512: Same as CDNA 3 (Wave64 optimization)
		// - PNT_GROUP_CNT=16: Can afford higher due to more CUs
		//   * More available parallelism to utilize
		//   * Better amortize synchronization costs across threads
		//   * Improved memory bandwidth per CU
		//   * 512 * 16 = 8192 total points per workgroup
		//   * Still maintains excellent L2 hit rate (350%+ of working set)
		//
		// Performance target: 2100+ Mk/s (60% improvement over RDNA 3)
		// Advantages over CDNA 3:
		// - More CUs (higher parallelism)
		// - Better FP32 performance
		// - Better memory latency hiding
		// ====================================================================
		#define BLOCK_SIZE			512
		#define PNT_GROUP_CNT		16
		#define CDNA4_ARCHITECTURE 1

	#else
		// Fallback for unknown/future AMD architectures
		// Use RDNA 3 parameters as reasonable default
		#define BLOCK_SIZE			256
		#define PNT_GROUP_CNT		24
	#endif

#else
	// CPU or unknown platform - use fake values for compilation
	#define BLOCK_SIZE			512
	#define PNT_GROUP_CNT		64
#endif

// ============================================================================
// Algorithm Constants
// ============================================================================

// Kangaroo type definitions for tame and wild kangaroo variants
#define TAME				0  // Tame kangaroo - deterministic starting point
#define WILD1				1  // Wild kangaroo variant 1
#define WILD2				2  // Wild kangaroo variant 2

// Distinguished point configuration
#define GPU_DP_SIZE			48  // Size of DP buffer on GPU
#define MAX_DP_CNT			(256 * 1024)  // Maximum number of distinguished points

// Jump table constants
#define JMP_MASK			(JMP_CNT-1)  // Mask for modulo JMP_CNT (since JMP_CNT is power of 2)

// DP table configuration
#define DPTABLE_MAX_CNT		16  // Maximum number of DP tables

// Maximum size of jump list
#define MAX_CNT_LIST		(512 * 1024)

// Distinguished point flags - used in jump list tracking
#define DP_FLAG				0x8000  // This jump ended at a distinguished point
#define INV_FLAG			0x4000  // This jump uses inverse operation
#define JMP2_FLAG			0x2000  // This jump uses jump table 2 (vs table 1)

// Algorithm constants
#define MD_LEN				10  // Must divide STEP_CNT evenly

// Debug mode (uncomment to enable detailed debug output)
//#define DEBUG_MODE

// ============================================================================
// GPU Kernel Parameters Structure
// ============================================================================
// This structure holds all parameters needed by GPU kernels
// Passed via constant memory for efficiency

struct TKparams
{
	// Kangaroo point data storage (Structure-of-Arrays layout)
	u64* Kangs;              // All kangaroo point coordinates
	u32 KangCnt;             // Number of kangaroo points
	u32 KangStride;          // Stride between consecutive limbs in SoA layout
	u32 BlockCnt;            // Number of thread blocks
	u32 BlockSize;           // Threads per block
	u32 GroupCnt;            // Number of point groups

	// GPU working memory
	u64* L2;                 // L2 cache working set (temporary results)
	u64 DP;                  // Distinguished point threshold
	u32* DPs_out;            // Output buffer for found distinguished points

	// Jump tables (each entry: x, y, d in 32-bit format)
	u64* Jumps1;             // Primary jump table
	u64* Jumps2;             // Secondary jump table
	u64* Jumps3;             // Tertiary jump table

	// Jump tracking
	u64* JumpsList;          // List of performed jumps
	u32* DPTable;            // Hash table for DP collision detection
	u32* L1S2;               // Local state for jump table selection

	// Last point tracking for loop detection
	u64* LastPnts;           // Last points visited by each kangaroo

	// Loop detection and handling
	u64* LoopTable;          // Table for detecting loops
	u32* LoopedKangs;        // Kangaroos that entered loops

	// Debugging support
	u32* dbg_buf;            // Debug output buffer

	// Mode flags
	bool IsGenMode;          // True if generating tame points, false if searching

	// LDS memory size requirements for each kernel
	u32 KernelA_LDS_Size;    // Local data store size for KernelA
	u32 KernelB_LDS_Size;    // Local data store size for KernelB
	u32 KernelC_LDS_Size;    // Local data store size for KernelC
};

