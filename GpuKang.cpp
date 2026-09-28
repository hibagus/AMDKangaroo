// RCKangaroo - AMD ROCm/HIP Port
// Original: (c) 2024 RetiredCoder (RC) - https://github.com/RetiredC
// License: GPLv3, see "LICENSE.TXT" file
// AMD Port: (c) 2025 Sirius437


#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#include "GpuKang.h"

hipError_t cuSetGpuParams(u64* _jmp2_table);
hipError_t CallGpuKernelGen(TKparams Kparams, const char*& failedOperation);
hipError_t CallGpuKernelABC(TKparams Kparams, const char*& failedOperation);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode
extern u32 gTotalErrors;

namespace
{
bool CheckHip(AMDGpuKang& gpu, hipError_t status, const char* operation)
{
	if (status == hipSuccess)
		return true;

	// Record only the first fatal HIP error for a worker. Cleanup calls may also
	// fail after a device error, but repeating those errors hides the root cause.
	if (!gpu.Failed)
	{
		printf("GPU %d: %s failed: %s (HIP error %d)\r\n",
			gpu.DeviceIndex, operation, hipGetErrorString(status), static_cast<int>(status));
		__sync_fetch_and_add(&gTotalErrors, 1U);
	}
	gpu.Failed = true;
	gpu.Stop();
	return false;
}

template <typename T>
void FreeGpuAllocation(AMDGpuKang& gpu, T*& allocation, const char* operation)
{
	if (!allocation)
		return;

	CheckHip(gpu, hipFree(allocation), operation);
	allocation = nullptr;
}

void ConfigureLaunchGeometry(TKparams& params, const AMDGpuProfile& profile, int processorCount)
{
	// More than one workgroup per processor can hide latency on CDNA, but it
	// also scales persistent state and temporary memory by the same multiplier.
	params.BlockCnt = processorCount * profile.GridMultiplier;
	params.BlockSize = profile.BlockSize;
	params.GroupCnt = profile.PointGroupCount;
}

template <typename Value>
Value Median(std::vector<Value> values)
{
	std::sort(values.begin(), values.end());
	const size_t middle = values.size() / 2;
	if (values.size() & 1)
		return values[middle];
	return (values[middle - 1] + values[middle]) / static_cast<Value>(2);
}

template <typename Value, typename Getter>
Value MedianSample(const std::vector<TGpuBenchmarkSample>& samples, Getter getter)
{
	std::vector<Value> values;
	values.reserve(samples.size());
	for (const TGpuBenchmarkSample& sample : samples)
		values.push_back(static_cast<Value>(getter(sample)));
	return Median(values);
}
} // namespace

// Helper function to convert AoS (Array of Structures) to SoA (Structure of Arrays)
// for coalesced GPU memory access
void ConvertAoStoSoA(TPointPriv* aos, u64* soa, int count)
{
	// AoS layout: Kang0[x0,x1,x2,x3,y0,y1,y2,y3,d0,d1,d2], Kang1[...], ...
	// SoA layout: [all x0][all x1][all x2][all x3][all y0][all y1][all y2][all y3][all d0][all d1][all d2]
	
	for (int i = 0; i < count; i++) {
		soa[i + 0 * count] = aos[i].x[0];  // All x0
		soa[i + 1 * count] = aos[i].x[1];  // All x1
		soa[i + 2 * count] = aos[i].x[2];  // All x2
		soa[i + 3 * count] = aos[i].x[3];  // All x3
		soa[i + 4 * count] = aos[i].y[0];  // All y0
		soa[i + 5 * count] = aos[i].y[1];  // All y1
		soa[i + 6 * count] = aos[i].y[2];  // All y2
		soa[i + 7 * count] = aos[i].y[3];  // All y3
		soa[i + 8 * count] = aos[i].priv[0];  // All d0
		soa[i + 9 * count] = aos[i].priv[1];  // All d1
		soa[i + 10 * count] = aos[i].priv[2]; // All d2
	}
}

int AMDGpuKang::CalcKangCnt()
{
	ConfigureLaunchGeometry(Kparams, *Profile, ProcessorCount);
	return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}

//executes in main thread
bool AMDGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	// Zero every allocation pointer so a partially prepared worker can release
	// exactly the resources that were acquired before an error.
	memset(&Kparams, 0, sizeof(Kparams));
	RndPnts = nullptr;
	DPs_out = nullptr;

	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	u64 total_mem = 0;
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;

	hipError_t err;
	err = hipSetDevice(DeviceIndex);
	if (!CheckHip(*this, err, "hipSetDevice during preparation"))
		return false;

	ConfigureLaunchGeometry(Kparams, *Profile, ProcessorCount);
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.KangStride = KangCnt;  // SoA layout: stride = KangCnt for coalesced access
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
	Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
	Kparams.IsGenMode = gGenMode;

//allocate gpu mem
	u64 size;
	if (Profile->UsesL2Workspace)
	{
		//L2	
		int L2size = Kparams.KangCnt * (3 * 32);
		total_mem += L2size;
		err = hipMalloc((void**)&Kparams.L2, L2size);
		if (!CheckHip(*this, err, "hipMalloc(L2 workspace)"))
			return false;
		size = L2size;
		if (size > persistingL2CacheMaxSize)
			size = persistingL2CacheMaxSize;
		// Note: HIP may not support all CUDA L2 cache features
		// Skipping hipDeviceSetLimit and stream attributes for now
		// TODO: Investigate AMD equivalent features
		/*
		err = hipDeviceSetLimit(hipLimitPersistingL2CacheSize, size);
		hipStreamAttrValue stream_attribute;                                                   
		stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
		stream_attribute.accessPolicyWindow.num_bytes = size;
		stream_attribute.accessPolicyWindow.hitRatio = 1.0;
		stream_attribute.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
		stream_attribute.accessPolicyWindow.missProp = hipAccessPropertyStreaming;
		err = hipStreamSetAttribute(NULL, hipStreamAttributeAccessPolicyWindow, &stream_attribute);
		if (err != hipSuccess)
		{
			printf("GPU %d, hipStreamSetAttribute failed: %s\n", DeviceIndex, hipGetErrorString(err));
			return false;
		}
		*/
	}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.DPs_out, size);
	if (!CheckHip(*this, err, "hipMalloc(DPs_out)"))
		return false;

	size = KangCnt * 96;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.Kangs, size);
	if (!CheckHip(*this, err, "hipMalloc(Kangs)"))
		return false;

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (!CheckHip(*this, err, "hipMalloc(Jumps1)"))
		return false;

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (!CheckHip(*this, err, "hipMalloc(Jumps2)"))
		return false;

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (!CheckHip(*this, err, "hipMalloc(Jumps3)"))
		return false;

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.JumpsList, size);
	if (!CheckHip(*this, err, "hipMalloc(JumpsList)"))
		return false;

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = hipMalloc((void**)&Kparams.DPTable, size);
	if (!CheckHip(*this, err, "hipMalloc(DPTable)"))
		return false;

	size = Kparams.BlockCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.L1S2, size);
	if (!CheckHip(*this, err, "hipMalloc(L1S2)"))
		return false;

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LastPnts, size);
	if (!CheckHip(*this, err, "hipMalloc(LastPnts)"))
		return false;

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LoopTable, size);
	if (!CheckHip(*this, err, "hipMalloc(LoopTable)"))
		return false;

	total_mem += 1024;
	err = hipMalloc((void**)&Kparams.dbg_buf, 1024);
	if (!CheckHip(*this, err, "hipMalloc(dbg_buf)"))
		return false;

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LoopedKangs, size);
	if (!CheckHip(*this, err, "hipMalloc(LoopedKangs)"))
		return false;

	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = hipMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, hipMemcpyHostToDevice);
	free(buf);
	if (!CheckHip(*this, err, "hipMemcpy(Jumps1 to device)"))
		return false;

//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = hipMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, hipMemcpyHostToDevice);
	free(buf);
	if (!CheckHip(*this, err, "hipMemcpy(Jumps2 to device)"))
		return false;


	err = cuSetGpuParams(jmp2_table);
	free(jmp2_table);
	if (!CheckHip(*this, err, "hipMemcpyToSymbol(jmp2_table)"))
		return false;
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = hipMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, hipMemcpyHostToDevice);
	free(buf);
	if (!CheckHip(*this, err, "hipMemcpy(Jumps3 to device)"))
		return false;


	printf("GPU %d: allocated %llu MB, %d kangaroos; %s launch: %u blocks x %u threads x %u groups\r\n",
		DeviceIndex, total_mem / (1024 * 1024), KangCnt, Profile->Name,
		Kparams.BlockCnt, Kparams.BlockSize, Kparams.GroupCnt);
	return true;
}

void AMDGpuKang::Release()
{
	free(RndPnts);
	RndPnts = nullptr;
	free(DPs_out);
	DPs_out = nullptr;
	FreeGpuAllocation(*this, Kparams.LoopedKangs, "hipFree(LoopedKangs)");
	FreeGpuAllocation(*this, Kparams.dbg_buf, "hipFree(dbg_buf)");
	FreeGpuAllocation(*this, Kparams.LoopTable, "hipFree(LoopTable)");
	FreeGpuAllocation(*this, Kparams.LastPnts, "hipFree(LastPnts)");
	FreeGpuAllocation(*this, Kparams.L1S2, "hipFree(L1S2)");
	FreeGpuAllocation(*this, Kparams.DPTable, "hipFree(DPTable)");
	FreeGpuAllocation(*this, Kparams.JumpsList, "hipFree(JumpsList)");
	FreeGpuAllocation(*this, Kparams.Jumps3, "hipFree(Jumps3)");
	FreeGpuAllocation(*this, Kparams.Jumps2, "hipFree(Jumps2)");
	FreeGpuAllocation(*this, Kparams.Jumps1, "hipFree(Jumps1)");
	FreeGpuAllocation(*this, Kparams.Kangs, "hipFree(Kangs)");
	FreeGpuAllocation(*this, Kparams.DPs_out, "hipFree(DPs_out)");
	if (Profile->UsesL2Workspace)
		FreeGpuAllocation(*this, Kparams.L2, "hipFree(L2 workspace)");
}

void AMDGpuKang::Stop()
{
	StopFlag = true;
}

void AMDGpuKang::GenerateRndDistances()
{
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		if (i < KangCnt / 3)
			d.RndBits(Range - 4); //TAME kangs
		else
		{
			d.RndBits(Range - 1);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

bool AMDGpuKang::Start()
{
	if (Failed)
		return false;

	hipError_t err;
	err = hipSetDevice(DeviceIndex);
	if (!CheckHip(*this, err, "hipSetDevice during initialization"))
		return false;

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
	PntB = PntA;
	PntB.y.NegModP();

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	GenerateRndDistances();
/* 
	//we can calc start points on CPU
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);
		d.data[3] = 0;
		d.data[4] = 0;
		EcPoint p = ec.MultiplyG(d);
		memcpy(RndPnts[i].x, p.x.data, 32);
		memcpy(RndPnts[i].y, p.y.data, 32);
	}
	for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntA);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntB);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	//copy to gpu - convert AoS to SoA for coalesced access
	u64* Kangs_SoA = (u64*)malloc(KangCnt * 96);
	ConvertAoStoSoA(RndPnts, Kangs_SoA, KangCnt);
	err = hipMemcpy(Kparams.Kangs, Kangs_SoA, KangCnt * 96, hipMemcpyHostToDevice);
	free(Kangs_SoA);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
/**/
	//but it's faster to calc then on GPU
	u8 buf_PntA[64], buf_PntB[64];
	PntA.SaveToBuffer64(buf_PntA);
	PntB.SaveToBuffer64(buf_PntB);
	for (int i = 0; i < KangCnt; i++)
	{
		if (i < KangCnt / 3)
			memset(RndPnts[i].x, 0, 64);
		else
			if (i < 2 * KangCnt / 3)
				memcpy(RndPnts[i].x, buf_PntA, 64);
			else
				memcpy(RndPnts[i].x, buf_PntB, 64);
	}
	//copy to gpu - convert AoS to SoA for coalesced access
	u64* Kangs_SoA2 = (u64*)malloc(KangCnt * 96);
	ConvertAoStoSoA(RndPnts, Kangs_SoA2, KangCnt);
	err = hipMemcpy(Kparams.Kangs, Kangs_SoA2, KangCnt * 96, hipMemcpyHostToDevice);
	free(Kangs_SoA2);
	if (!CheckHip(*this, err, "hipMemcpy(initial kangaroo state to device)"))
		return false;

	const char* failedOperation = nullptr;
	err = CallGpuKernelGen(Kparams, failedOperation);
	if (!CheckHip(*this, err, failedOperation))
		return false;

	err = hipMemset(Kparams.L1S2, 0, Kparams.BlockCnt * Kparams.BlockSize * 8);
	if (!CheckHip(*this, err, "hipMemset(loop flags)"))
		return false;
	err = hipMemset(Kparams.dbg_buf, 0, 1024);
	if (!CheckHip(*this, err, "hipMemset(debug buffer)"))
		return false;
	err = hipMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	if (!CheckHip(*this, err, "hipMemset(loop history)"))
		return false;
	return true;
}

#ifdef DEBUG_MODE
int AMDGpuKang::Dbg_CheckKangs()
{
	int kang_size = KangCnt * 96;
	u64* kangs = (u64*)malloc(kang_size);
	hipError_t err = hipMemcpy(kangs, Kparams.Kangs, kang_size, hipMemcpyDeviceToHost);
	if (!CheckHip(*this, err, "hipMemcpy(debug kangaroo state to host)"))
	{
		free(kangs);
		return -1;
	}
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &kangs[i * 12 + 8], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			if (i < 2 * KangCnt / 3)
				p = ec.AddPoints(PntA, p);
			else
				p = ec.AddPoints(PntB, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	return res;
}
#endif

bool AMDGpuKang::Benchmark(const TGpuBenchmarkConfig& config,
	TGpuBenchmarkResult& result)
{
	result = {};
	if (!config.TimedIterations)
	{
		printf("GPU %d: benchmark requires at least one timed iteration\r\n",
			DeviceIndex);
		return false;
	}

	if (!Start())
	{
		Release();
		return false;
	}

	hipEvent_t startEvent = nullptr;
	hipEvent_t stopEvent = nullptr;
	bool success = CheckHip(*this, hipEventCreate(&startEvent),
		"hipEventCreate(benchmark start)");
	if (success)
		success = CheckHip(*this, hipEventCreate(&stopEvent),
			"hipEventCreate(benchmark stop)");

	const u32 totalIterations =
		config.WarmupIterations + config.TimedIterations;
	result.JumpsPerIteration = static_cast<u64>(KangCnt) * STEP_CNT;
	result.Samples.reserve(config.TimedIterations);

	for (u32 iteration = 0; success && iteration < totalIterations; iteration++)
	{
		TGpuBenchmarkSample sample{};
		const auto wallStart = std::chrono::steady_clock::now();

		success = CheckHip(*this, hipEventRecord(startEvent),
			"hipEventRecord(clear start)");
		if (success)
			success = CheckHip(*this, hipMemset(Kparams.DPs_out, 0, 4),
				"hipMemset(benchmark distinguished-point count)");
		if (success)
			success = CheckHip(*this,
				hipMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32)),
				"hipMemset(benchmark distinguished-point counters)");
		if (success)
			success = CheckHip(*this, hipMemset(Kparams.LoopedKangs, 0, 8),
				"hipMemset(benchmark looped-kangaroo count)");
		if (success)
			success = CheckHip(*this, hipEventRecord(stopEvent),
				"hipEventRecord(clear stop)");
		if (success)
			success = CheckHip(*this, hipEventSynchronize(stopEvent),
				"hipEventSynchronize(clear)");
		if (success)
			success = CheckHip(*this,
				hipEventElapsedTime(&sample.ClearMilliseconds,
					startEvent, stopEvent),
				"hipEventElapsedTime(clear)");
		if (!success)
			break;

		const char* failedOperation = nullptr;
		const hipError_t kernelStatus = CallGpuKernelABCTimed(
			Kparams, startEvent, stopEvent, sample.Kernels, failedOperation);
		success = CheckHip(*this, kernelStatus,
			failedOperation ? failedOperation : "timed kernel sequence");
		if (!success)
			break;

		success = CheckHip(*this, hipEventRecord(startEvent),
			"hipEventRecord(transfer start)");
		int pointCount = 0;
		if (success)
			success = CheckHip(*this,
				hipMemcpy(&pointCount, Kparams.DPs_out, sizeof(pointCount),
					hipMemcpyDeviceToHost),
				"hipMemcpy(benchmark distinguished-point count)");
		if (pointCount < 0)
			pointCount = 0;
		else if (pointCount >= MAX_DP_CNT)
			pointCount = MAX_DP_CNT;
		if (success && pointCount)
			success = CheckHip(*this,
				hipMemcpy(DPs_out, Kparams.DPs_out + 4,
					static_cast<size_t>(pointCount) * GPU_DP_SIZE,
					hipMemcpyDeviceToHost),
				"hipMemcpy(benchmark distinguished points)");
		if (success)
			success = CheckHip(*this,
				hipMemcpy(dbg, Kparams.dbg_buf, sizeof(dbg),
					hipMemcpyDeviceToHost),
				"hipMemcpy(benchmark debug counters)");
		u32 loopedCount = 0;
		if (success)
			success = CheckHip(*this,
				hipMemcpy(&loopedCount, Kparams.LoopedKangs,
					sizeof(loopedCount), hipMemcpyDeviceToHost),
				"hipMemcpy(benchmark looped-kangaroo count)");
		if (success)
			success = CheckHip(*this, hipEventRecord(stopEvent),
				"hipEventRecord(transfer stop)");
		if (success)
			success = CheckHip(*this, hipEventSynchronize(stopEvent),
				"hipEventSynchronize(transfer)");
		if (success)
			success = CheckHip(*this,
				hipEventElapsedTime(&sample.TransferMilliseconds,
					startEvent, stopEvent),
				"hipEventElapsedTime(transfer)");
		if (!success)
			break;

		const auto wallStop = std::chrono::steady_clock::now();
		sample.TotalMilliseconds =
			std::chrono::duration<double, std::milli>(
				wallStop - wallStart).count();
		sample.DistinguishedPointCount = static_cast<u32>(pointCount);
		sample.LoopedKangarooCount = loopedCount;

		if (iteration >= config.WarmupIterations)
		{
			result.TimedDurationSeconds += sample.TotalMilliseconds / 1000.0;
			result.Samples.push_back(sample);
		}
	}

	if (stopEvent)
		success = CheckHip(*this, hipEventDestroy(stopEvent),
			"hipEventDestroy(benchmark stop)") && success;
	if (startEvent)
		success = CheckHip(*this, hipEventDestroy(startEvent),
			"hipEventDestroy(benchmark start)") && success;

	Release();
	if (!success || result.Samples.size() != config.TimedIterations)
		return false;

	// Report medians rather than means so a transient dispatch or host delay
	// cannot make a weak optimization look better or worse than it is.
	result.Median.ClearMilliseconds = MedianSample<float>(
		result.Samples,
		[](const TGpuBenchmarkSample& value) { return value.ClearMilliseconds; });
	result.Median.Kernels.KernelA_Milliseconds = MedianSample<float>(
		result.Samples,
		[](const TGpuBenchmarkSample& value)
		{
			return value.Kernels.KernelA_Milliseconds;
		});
	result.Median.Kernels.KernelB_Milliseconds = MedianSample<float>(
		result.Samples,
		[](const TGpuBenchmarkSample& value)
		{
			return value.Kernels.KernelB_Milliseconds;
		});
	result.Median.Kernels.KernelC_Milliseconds = MedianSample<float>(
		result.Samples,
		[](const TGpuBenchmarkSample& value)
		{
			return value.Kernels.KernelC_Milliseconds;
		});
	result.Median.TransferMilliseconds = MedianSample<float>(
		result.Samples,
		[](const TGpuBenchmarkSample& value)
		{
			return value.TransferMilliseconds;
		});
	result.Median.TotalMilliseconds = MedianSample<double>(
		result.Samples,
		[](const TGpuBenchmarkSample& value) { return value.TotalMilliseconds; });
	result.Median.DistinguishedPointCount = MedianSample<u32>(
		result.Samples,
		[](const TGpuBenchmarkSample& value)
		{
			return value.DistinguishedPointCount;
		});
	result.Median.LoopedKangarooCount = MedianSample<u32>(
		result.Samples,
		[](const TGpuBenchmarkSample& value)
		{
			return value.LoopedKangarooCount;
		});
	return true;
}

//executes in separate thread
void AMDGpuKang::Execute()
{
	if (Failed)
	{
		// Preparation can fail after some allocations. Select the owning device
		// before releasing the successfully allocated subset.
		if (CheckHip(*this, hipSetDevice(DeviceIndex), "hipSetDevice during failed-worker cleanup"))
			Release();
		return;
	}

	if (!Start())
	{
		Release();
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	hipError_t err;
	const char* failedOperation = nullptr;
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();

		err = hipMemset(Kparams.DPs_out, 0, 4);
		if (!CheckHip(*this, err, "hipMemset(distinguished-point output count)"))
			break;
		err = hipMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
		if (!CheckHip(*this, err, "hipMemset(distinguished-point counters)"))
			break;
		err = hipMemset(Kparams.LoopedKangs, 0, 8);
		if (!CheckHip(*this, err, "hipMemset(looped-kangaroo count)"))
			break;

		err = CallGpuKernelABC(Kparams, failedOperation);
		if (!CheckHip(*this, err, failedOperation))
			break;

		int cnt = 0;
		err = hipMemcpy(&cnt, Kparams.DPs_out, 4, hipMemcpyDeviceToHost);
		if (!CheckHip(*this, err, "hipMemcpy(distinguished-point count to host)"))
			break;

		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", DeviceIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = hipMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, hipMemcpyDeviceToHost);
			if (!CheckHip(*this, err, "hipMemcpy(distinguished points to host)"))
				break;
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

		err = hipMemcpy(dbg, Kparams.dbg_buf, 1024, hipMemcpyDeviceToHost);
		if (!CheckHip(*this, err, "hipMemcpy(debug counters to host)"))
			break;

		u32 lcnt = 0;
		err = hipMemcpy(&lcnt, Kparams.LoopedKangs, 4, hipMemcpyDeviceToHost);
		if (!CheckHip(*this, err, "hipMemcpy(looped-kangaroo count to host)"))
			break;
		//printf("GPU %d, Looped: %d\r\n", DeviceIndex, lcnt);

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", DeviceIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt < 0)
				break;
			if (corr_cnt > 0)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", DeviceIndex, corr_cnt);
				__sync_fetch_and_add(&gTotalErrors, 1U);
				Failed = true;
				StopFlag = true;
				break;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", DeviceIndex);
		}
		iter++;
#endif
	}

	Release();
}

int AMDGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}