// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once 

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;



#define MAX_GPU_CNT			32

// Production builds process 1,000 jumps per launch. Tests may override this at
// compile time to inspect exact state after individual jumps; KernelB retains
// its faster full-batch path whenever the value is divisible by MD_LEN.
#ifndef STEP_CNT
#define STEP_CNT			1000
#endif

#define JMP_CNT				512

// Preserve the upstream CUDA modes; HIP targets use explicit AMD profiles below.
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ < 890
		#define OLD_GPU
	#endif
	#ifdef OLD_GPU
		#define BLOCK_SIZE			512
		//can be 8, 16, 24, 32, 40, 48, 56, 64
		#define PNT_GROUP_CNT		64	
	#else
		#define BLOCK_SIZE			512
		//can be 8, 16, 24, 32
		#define PNT_GROUP_CNT		24
	#endif
#elif defined(__HIP_PLATFORM_AMD__)
	// Conservative shared geometry for gfx1100, gfx942, and gfx950 bring-up.
	// Architecture-specific values will be introduced only after correctness tests
	// and occupancy measurements justify them.
	#define BLOCK_SIZE			256
	#define PNT_GROUP_CNT		24
#else //CPU, fake values
	#define BLOCK_SIZE			512
	#define PNT_GROUP_CNT		64
#endif

// kang type
#define TAME				0  // Tame kangs
#define WILD1				1  // Wild kangs1 
#define WILD2				2  // Wild kangs2

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

#define JMP_MASK			(JMP_CNT-1)

#define DPTABLE_MAX_CNT		16

#define MAX_CNT_LIST		(512 * 1024)

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG			0x2000

#define MD_LEN				10

// LoopedKangs entries keep the kangaroo index in the low 28 bits and the
// LastPnts history slot in the high four bits. MD_LEN must therefore remain
// no greater than 16 unless this packing format is changed.
#define LOOPED_KANG_INDEX_MASK		0x0FFFFFFFu
#define LOOPED_KANG_HISTORY_SHIFT	28

//#define DEBUG_MODE

// Flat device-buffer layouts shared by the three runtime kernels. The layouts
// favor coalesced accesses, so most arrays are structures of arrays (SoA)
// rather than conventional per-kangaroo records.
struct TKparams
{
	// Primary state: [12 planes][KangStride entries]. Planes 0..3 are X,
	// 4..7 are Y, 8..10 are the signed 192-bit distance, and plane 11 is
	// reserved padding. Only the first KangCnt entries of each plane are live.
	u64* Kangs;
	u32 KangCnt;
	u32 KangStride;

	// Launch geometry. KangCnt is normally BlockCnt * BlockSize * GroupCnt.
	u32 BlockCnt;
	u32 BlockSize;
	u32 GroupCnt;

	// KernelA scratch: three 256-bit values per kangaroo (X, Y, and the
	// batched-inversion workspace). The buffer is not persistent between runs.
	u64* L2;
	u64 DP;

	// Distinguished-point output: a 16-byte header followed by MAX_DP_CNT
	// 48-byte records. Header word 0 is the record count. Each record contains
	// X[0..1], a signed 192-bit distance, a type word, and four padding bytes.
	u32* DPs_out;

	// Jump tables: [JMP_CNT][12 u64] = X[4], Y[4], distance[4]. The distance
	// is signed 192-bit data in the first three limbs; the fourth limb pads the
	// record for aligned vector loads.
	u64* Jumps1;
	u64* Jumps2;
	u64* Jumps3;

	// KernelA -> KernelB jump stream. Logically indexed as
	// [block][step][kangaroo-within-block], but physically wave-interleaved for
	// coalescing. Each 16-bit descriptor stores a 9-bit JMP_MASK index plus
	// JMP2_FLAG, INV_FLAG, and DP_FLAG.
	u64* JumpsList;

	// KernelA -> KernelB distinguished-point staging. The first KangCnt u32s
	// are packed producer/consumer counters (low/high 16 bits). They are
	// followed by [kangaroo][DPTABLE_MAX_CNT][4 u32] low-128-bit X values.
	u32* DPTable;

	// One persistent loop-state bitfield per [block][thread]. A bit selects
	// jump table 2 for its point group. Allocation retains an eight-byte slot
	// for compatibility with configurations that use up to 64 groups.
	u32* L1S2;

	// KernelA -> KernelC point history, logically
	// [MD_LEN][X/Y][4 limbs][kangaroo], with kangaroos interleaved for
	// coalesced loads. KernelC uses the selected slot to escape a loop.
	u64* LastPnts;

	// KernelB's distance history: [block][group][MD_LEN][thread]. THREAD_X is
	// the innermost index so every lane owns an independent loop history.
	u64* LoopTable;

	// Debug histogram indexed by detected loop length.
	u32* dbg_buf;

	// KernelB -> KernelC queue. Word 0 is the producer count, word 1 is the
	// consumer cursor, and later words pack a low-28-bit kangaroo index with a
	// high-4-bit LastPnts slot.
	u32* LoopedKangs;
	bool IsGenMode; // Tame-kangaroo generation mode.

	// Dynamically allocated local-data-share bytes for each kernel.
	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;	
};

