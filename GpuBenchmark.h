// Reproducible GPU benchmark data shared by the production host and HIP code.

#pragma once

#include <vector>

#include <hip/hip_runtime.h>

#include "defs.h"

struct TGpuKernelTimings
{
	float KernelA_Milliseconds = 0.0f;
	float KernelB_Milliseconds = 0.0f;
	float KernelC_Milliseconds = 0.0f;
};

struct TGpuBenchmarkConfig
{
	u32 WarmupIterations = 0;
	u32 TimedIterations = 0;
	u64 Seed = 0;
};

struct TGpuBenchmarkSample
{
	float ClearMilliseconds = 0.0f;
	TGpuKernelTimings Kernels;
	float TransferMilliseconds = 0.0f;
	double TotalMilliseconds = 0.0;
	u32 DistinguishedPointCount = 0;
	u32 LoopedKangarooCount = 0;
};

struct TGpuBenchmarkResult
{
	u64 JumpsPerIteration = 0;
	std::vector<TGpuBenchmarkSample> Samples;
	TGpuBenchmarkSample Median;
	double TimedDurationSeconds = 0.0;
};

// Reuses a caller-owned event pair so event creation and destruction are not
// charged to every timed iteration.
hipError_t CallGpuKernelABCTimed(TKparams params, hipEvent_t startEvent,
	hipEvent_t stopEvent, TGpuKernelTimings& timings,
	const char*& failedOperation);
