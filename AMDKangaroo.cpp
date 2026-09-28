// RCKangaroo - AMD ROCm/HIP Port
// Original: (c) 2024 RetiredCoder (RC) - https://github.com/RetiredC
// AMD Port: (c) 2025 Sirius437
// License: GPLv3, see "LICENSE.TXT" file


#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"
#include "GpuBenchmarkReport.h"


EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

AMDGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;
bool gKernelBenchmark;
bool gBenchmarkOptionSeen;
TGpuBenchmarkConfig gBenchmarkConfig;
char gBenchmarkOutput[1024];

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

namespace
{
bool CheckHipInitialization(hipError_t status, const char* operation, int deviceIndex = -1)
{
	if (status == hipSuccess)
		return true;

	if (deviceIndex >= 0)
		printf("GPU %d: %s failed: %s (HIP error %d)\r\n",
			deviceIndex, operation, hipGetErrorString(status), static_cast<int>(status));
	else
		printf("%s failed: %s (HIP error %d)\r\n",
			operation, hipGetErrorString(status), static_cast<int>(status));
	return false;
}

bool ParseUnsigned(const char* text, u32 minimum, u32 maximum, u32& value)
{
	errno = 0;
	char* end = nullptr;
	const unsigned long parsed = std::strtoul(text, &end, 10);
	if (errno || end == text || *end != '\0'
		|| parsed < minimum || parsed > maximum)
		return false;
	value = static_cast<u32>(parsed);
	return true;
}

bool ParseSeed(const char* text, u64& value)
{
	errno = 0;
	char* end = nullptr;
	const unsigned long long parsed = std::strtoull(text, &end, 0);
	if (errno || end == text || *end != '\0' || text[0] == '-')
		return false;
	value = static_cast<u64>(parsed);
	return true;
}

bool RequireOptionValue(int argumentIndex, int argumentCount, const char* option)
{
	if (argumentIndex < argumentCount)
		return true;
	printf("error: missed value after %s option\r\n", option);
	return false;
}
} // namespace

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	if (!CheckHipInitialization(hipGetDeviceCount(&gcnt), "hipGetDeviceCount"))
		return;
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

	if (!gcnt)
		return;

	int driverVersion = 0;
	int runtimeVersion = 0;
	if (!CheckHipInitialization(hipDriverGetVersion(&driverVersion), "hipDriverGetVersion"))
		return;
	if (!CheckHipInitialization(hipRuntimeGetVersion(&runtimeVersion), "hipRuntimeGetVersion"))
		return;

	// HIP encodes versions as major * 10,000,000 + minor * 100,000 + patch.
	printf("HIP devices: %d, HIP driver/runtime: %d.%d.%d/%d.%d.%d\r\n", gcnt,
		driverVersion / 10000000, (driverVersion / 100000) % 100, driverVersion % 100000,
		runtimeVersion / 10000000, (runtimeVersion / 100000) % 100, runtimeVersion % 100000);

	for (int i = 0; i < gcnt; i++)
	{
		if (!gGPUs_Mask[i])
			continue;

		if (!CheckHipInitialization(hipSetDevice(i), "hipSetDevice during discovery", i))
			continue;

		hipDeviceProp_t deviceProp;
		if (!CheckHipInitialization(
				hipGetDeviceProperties(&deviceProp, i), "hipGetDeviceProperties", i))
			continue;
		const AMDGpuProfile* profile = FindAMDGpuProfile(deviceProp.gcnArchName);
		if (!profile)
		{
			printf("GPU %d: unsupported architecture '%s'; expected gfx1100, gfx942, or gfx950\r\n",
				i, deviceProp.gcnArchName);
			continue;
		}

		if (std::strcmp(profile->Name, GetCompiledGpuArchitecture()) != 0)
		{
			printf("GPU %d: %s device cannot use this %s binary; skipping\r\n",
				i, profile->Name, GetCompiledGpuArchitecture());
			continue;
		}

		if (profile->BlockSize > deviceProp.maxThreadsPerBlock)
		{
			printf("GPU %d: profile requires %d threads per block, but device allows %d; skipping\r\n",
				i, profile->BlockSize, deviceProp.maxThreadsPerBlock);
			continue;
		}

		// KernelC has the largest dynamic LDS request among the current kernels.
		const size_t requiredLdsBytes = 96ULL * JMP_CNT;
		if (requiredLdsBytes > deviceProp.sharedMemPerBlock)
		{
			printf("GPU %d: kernels require %zu KB LDS, but device allows %zu KB; skipping\r\n",
				i, requiredLdsBytes / 1024, deviceProp.sharedMemPerBlock / 1024);
			continue;
		}

		const int actualCUs = deviceProp.multiProcessorCount * profile->ComputeUnitsPerProcessor;
		printf("GPU %d: %s, %.2f GB, architecture %s, wave %d, "
			"scheduler units: %d %s, physical CUs: %d, "
			"LDS/block: %zu KB, LDS/processor: %zu KB, PCI %d, L2 size: %d KB\r\n",
			i, deviceProp.name, static_cast<double>(deviceProp.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
			profile->Name, deviceProp.warpSize, deviceProp.multiProcessorCount,
			profile->ProcessorUnitName, actualCUs, deviceProp.sharedMemPerBlock / 1024,
			deviceProp.maxSharedMemoryPerMultiProcessor / 1024, deviceProp.pciBusID,
			deviceProp.l2CacheSize / 1024);

		if (!CheckHipInitialization(
				hipSetDeviceFlags(hipDeviceScheduleBlockingSync), "hipSetDeviceFlags", i))
			continue;

		GpuKangs[GpuCnt] = new AMDGpuKang();
		GpuKangs[GpuCnt]->DeviceIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->ProcessorCount = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->Profile = profile;
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	AMDGpuKang* Kang = (AMDGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	AMDGpuKang* Kang = (AMDGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		csAddPoints.Leave();
		printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
		return;
	}
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	PntTotalOps += ops_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
}


void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

	for (int i = 0; i < cnt; i++)
	{
		DBRec nrec;
		u8* p = pPntList2 + i * GPU_DP_SIZE;
		memcpy(nrec.x, p, 12);
		memcpy(nrec.d, p + 16, 22);
		nrec.type = gGenMode ? TAME : p[40];

		DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
		if (gGenMode)
			continue;
		if (pref)
		{
			//in db we dont store first 3 bytes so restore them
			DBRec tmp_pref;
			memcpy(&tmp_pref, &nrec, 3);
			memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
			pref = &tmp_pref;

			if (pref->type == nrec.type)
			{
				if (pref->type == TAME)
					continue;

				//if it's wild, we can find the key from the same type if distances are different
				if (*(u64*)pref->d == *(u64*)nrec.d)
					continue;
				//else
				//	ToLog("key found by same wild");
			}

			EcInt w, t;
			int TameType, WildType;
			if (pref->type != TAME)
			{
				memcpy(w.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = nrec.type;
				WildType = pref->type;
			}
			else
			{
				memcpy(w.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = TAME;
				WildType = nrec.type;
			}

			bool res = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
			if (!res)
			{
				bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
				if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
					;// ToLog("W1 and W2 collides in mirror");
				else
				{
					printf("Collision Error\r\n");
					gTotalErrors++;
				}
				continue;
			}
			gSolved = true;
			break;
		}
	}
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
	for (int i = 0; i <= MD_LEN; i++)
	{
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	int speed = GpuKangs[0]->GetStatsSpeed();
	for (int i = 1; i < GpuCnt; i++)
		speed += GpuKangs[i]->GetStatsSpeed();

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;
	if (speed)
		exp_sec = (u64)((exp_ops / 1000000) / speed); //in sec
	u64 exp_days = exp_sec / (3600 * 24);
	int exp_hours = (int)(exp_sec - exp_days * (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec - exp_days * (3600 * 24) - exp_hours * 3600) / 60;

	u64 sec = (GetTickCount64() - tm_start) / 1000;
	u64 days = sec / (3600 * 24);
	int hours = (int)(sec - days * (3600 * 24)) / 3600;
	int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;
	 
	printf("%sSpeed: %d MKeys/s, Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm/%llud:%02dh:%02dm\r\n", gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "), speed, gTotalErrors, db.GetBlockCnt()/1000, est_dps_cnt/1000, days, hours, min, exp_days, exp_hours, exp_min);
}

bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 14) || (DP > 60)) 
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}

	printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
	gIsOpsLimit = false;
	double MaxTotalOps = 0.0;
	if (gMax > 0)
	{
		MaxTotalOps = gMax * ops;
		double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
		ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
		ram_max /= (1024 * 1024 * 1024); //GB
		printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
	}

	u64 total_kangs = GpuKangs[0]->CalcKangCnt();
	for (int i = 1; i < GpuCnt; i++)
		total_kangs += GpuKangs[i]->CalcKangCnt();
	double path_single_kang = ops / total_kangs;	
	double DPs_per_kang = path_single_kang / dp_val;
	printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

	if (!gGenMode && gTamesFileName[0])
	{
		printf("load tames...\r\n");
		if (db.LoadFromFile(gTamesFileName))
		{
			printf("tames loaded\r\n");
			if (db.Header[0] != gRange)
			{
				printf("loaded tames have different range, they cannot be used, clear\r\n");
				db.Clear();
			}
		}
		else
			printf("tames loading failed\r\n");
	}

	SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;
//prepare jumps
	EcInt minjump, t;
	minjump.Set(1);
	minjump.ShiftLeft(Range / 2 + 3);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
	}
	SetRndSeed(GetTickCount64());

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(Range - 1);
	Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();
	Int_TameOffset.Set(1);
	Int_TameOffset.ShiftLeft(Range - 1);
	EcInt tt;
	tt.Set(1);
	tt.ShiftLeft(Range - 5); //half of tame range width
	Int_TameOffset.Sub(tt);
	gPntToSolve = PntToSolve;

//prepare GPUs
	for (int i = 0; i < GpuCnt; i++)
		if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
			GpuKangs[i]->Failed = true;

	u64 tm0 = GetTickCount64();
	printf("GPUs started...\r\n");

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT];
#else
	pthread_t thr_handles[MAX_GPU_CNT];
#endif

	u32 ThreadID;
	gSolved = false;
	ThrCnt = GpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
	while (!gSolved)
	{
		CheckNewPoints();
		Sleep(10);

		// A failed worker decrements ThrCnt before exiting. Stop waiting when no
		// GPU can make further progress, even if no collision was found.
		if (!ThrCnt)
		{
			printf("All GPU workers stopped before a solution was found.\r\n");
			break;
		}
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val);
			tm_stats = GetTickCount64();
		}

		if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
		{
			gIsOpsLimit = true;
			printf("Operations limit reached\r\n");
			break;
		}
	}

	printf("Stopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(10);
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}

	if (gIsOpsLimit)
	{
		if (gGenMode)
		{
			printf("saving tames...\r\n");
			db.Header[0] = gRange; 
			if (db.SaveToFile(gTamesFileName))
				printf("tames saved\r\n");
			else
				printf("tames saving failed\r\n");
		}
		db.Clear();
		return false;
	}

	if (!gSolved)
	{
		db.Clear();
		return false;
	}

	double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
	printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
	db.Clear();
	*pk_res = gPrivKey;
	return true;
}

bool RunKernelBenchmark()
{
	if (GpuCnt != 1)
	{
		printf("error: reproducible benchmarking requires exactly one selected "
			"GPU; use -gpu with one device index\r\n");
		return false;
	}
	if (!gRange)
		gRange = 78;
	if (!gDP)
		gDP = 16;

	AMDGpuKang& gpu = *GpuKangs[0];
	if (!CheckHipInitialization(hipSetDevice(gpu.DeviceIndex),
			"hipSetDevice for benchmark", gpu.DeviceIndex))
		return false;

	hipDeviceProp_t properties{};
	if (!CheckHipInitialization(
			hipGetDeviceProperties(&properties, gpu.DeviceIndex),
			"hipGetDeviceProperties for benchmark", gpu.DeviceIndex))
		return false;

	TGpuBenchmarkMetadata metadata;
	metadata.TimestampUtc = CurrentUtcTimestamp();
	metadata.DeviceIndex = gpu.DeviceIndex;
	metadata.DeviceName = properties.name;
	metadata.DeviceUuid = FormatGpuUuid(properties.uuid);
	metadata.Architecture = gpu.Profile->Name;
	if (!CheckHipInitialization(hipDriverGetVersion(&metadata.DriverVersion),
			"hipDriverGetVersion for benchmark", gpu.DeviceIndex)
		|| !CheckHipInitialization(hipRuntimeGetVersion(&metadata.RuntimeVersion),
			"hipRuntimeGetVersion for benchmark", gpu.DeviceIndex))
		return false;
	metadata.WaveSize = properties.warpSize;
	metadata.SchedulerUnits = properties.multiProcessorCount;
	metadata.PhysicalComputeUnits =
		properties.multiProcessorCount * gpu.Profile->ComputeUnitsPerProcessor;
	metadata.NominalCoreClockKHz = properties.clockRate;
	metadata.NominalMemoryClockKHz = properties.memoryClockRate;
	metadata.BlockSize = gpu.Profile->BlockSize;
	metadata.PointGroupCount = gpu.Profile->PointGroupCount;
	metadata.Range = gRange;
	metadata.DistinguishedPointBits = gDP;
	metadata.StepCount = STEP_CNT;

	std::string telemetryWarning;
	if (!CollectGpuTelemetry(properties, metadata.Before, telemetryWarning))
		printf("Benchmark telemetry before run unavailable: %s\r\n",
			telemetryWarning.c_str());

	// Derive every part of the workload from the recorded seed. Repeating the
	// command therefore creates the same target point, jump tables, and initial
	// kangaroo distances.
	SetRndSeed(gBenchmarkConfig.Seed);
	EcInt benchmarkPrivateKey;
	benchmarkPrivateKey.RndBits(gRange);
	if (benchmarkPrivateKey.IsZero())
		benchmarkPrivateKey.Set(1);
	EcPoint pointToSolve = ec.MultiplyG(benchmarkPrivateKey);

	EcInt minimumJump;
	EcInt randomOffset;
	minimumJump.Set(1);
	minimumJump.ShiftLeft(gRange / 2 + 3);
	for (int index = 0; index < JMP_CNT; index++)
	{
		EcJumps1[index].dist = minimumJump;
		randomOffset.RndMax(minimumJump);
		EcJumps1[index].dist.Add(randomOffset);
		EcJumps1[index].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
		EcJumps1[index].p = ec.MultiplyG(EcJumps1[index].dist);
	}

	minimumJump.Set(1);
	minimumJump.ShiftLeft(gRange - 10);
	for (int index = 0; index < JMP_CNT; index++)
	{
		EcJumps2[index].dist = minimumJump;
		randomOffset.RndMax(minimumJump);
		EcJumps2[index].dist.Add(randomOffset);
		EcJumps2[index].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
		EcJumps2[index].p = ec.MultiplyG(EcJumps2[index].dist);
	}

	minimumJump.Set(1);
	minimumJump.ShiftLeft(gRange - 12);
	for (int index = 0; index < JMP_CNT; index++)
	{
		EcJumps3[index].dist = minimumJump;
		randomOffset.RndMax(minimumJump);
		EcJumps3[index].dist.Add(randomOffset);
		EcJumps3[index].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
		EcJumps3[index].p = ec.MultiplyG(EcJumps3[index].dist);
	}

	// Start() consumes the shared RNG when it creates initial distances. Reset
	// it after jump generation so the starting state is independent of later
	// changes to jump-table construction.
	SetRndSeed(gBenchmarkConfig.Seed ^ 0x4B414E4741524F4FULL);
	const bool prepared = gpu.Prepare(pointToSolve, gRange, gDP,
		EcJumps1, EcJumps2, EcJumps3);
	// Benchmark() also releases partially allocated state when Prepare() fails,
	// so always call it before returning an initialization error.
	TGpuBenchmarkResult result;
	const bool benchmarked = gpu.Benchmark(gBenchmarkConfig, result);
	if (!prepared || !benchmarked)
		return false;

	telemetryWarning.clear();
	if (!CollectGpuTelemetry(properties, metadata.After, telemetryWarning))
		printf("Benchmark telemetry after run unavailable: %s\r\n",
			telemetryWarning.c_str());

	PrintGpuBenchmarkSummary(metadata, gBenchmarkConfig, result);
	if (!AppendGpuBenchmarkCsv(
			gBenchmarkOutput, metadata, gBenchmarkConfig, result))
		return false;
	printf("Benchmark samples appended to %s\r\n", gBenchmarkOutput);
	return true;
}

bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if (strcmp(argument, "-benchmark") == 0)
		{
			gKernelBenchmark = true;
		}
		else
		if (strcmp(argument, "-bench-warmup") == 0)
		{
			gBenchmarkOptionSeen = true;
			if (!RequireOptionValue(ci, argc, argument)
				|| !ParseUnsigned(argv[ci], 0, 1000,
					gBenchmarkConfig.WarmupIterations))
			{
				printf("error: invalid value for -bench-warmup option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-bench-iterations") == 0)
		{
			gBenchmarkOptionSeen = true;
			if (!RequireOptionValue(ci, argc, argument)
				|| !ParseUnsigned(argv[ci], 1, 1000,
					gBenchmarkConfig.TimedIterations))
			{
				printf("error: invalid value for -bench-iterations option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-bench-seed") == 0)
		{
			gBenchmarkOptionSeen = true;
			if (!RequireOptionValue(ci, argc, argument)
				|| !ParseSeed(argv[ci], gBenchmarkConfig.Seed))
			{
				printf("error: invalid value for -bench-seed option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-bench-output") == 0)
		{
			gBenchmarkOptionSeen = true;
			if (!RequireOptionValue(ci, argc, argument)
				|| !argv[ci][0]
				|| strlen(argv[ci]) >= sizeof(gBenchmarkOutput))
			{
				printf("error: invalid value for -bench-output option\r\n");
				return false;
			}
			strcpy(gBenchmarkOutput, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 14) || (val > 60))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-tames") == 0)
		{
			strcpy(gTamesFileName, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-max") == 0)
		{
			double val = atof(argv[ci]);
			ci++;
			if (val < 0.001)
			{
				printf("error: invalid value for -max option\r\n");
				return false;
			}
			gMax = val;
		}
		else
		{
			printf("error: unknown option %s\r\n", argument);
			return false;
		}
	}
	if (gBenchmarkOptionSeen && !gKernelBenchmark)
	{
		printf("error: -bench-* options require -benchmark\r\n");
		return false;
	}
	if (gKernelBenchmark)
	{
		if (!gBenchmarkOutput[0])
		{
			printf("error: -benchmark requires -bench-output FILE\r\n");
			return false;
		}
		if (!gPubKey.x.IsZero() || gTamesFileName[0] || gMax > 0.0
			|| gStartSet)
		{
			printf("error: -benchmark cannot be combined with -pubkey, "
				"-start, -tames, or -max\r\n");
			return false;
		}
	}
	if (!gPubKey.x.IsZero())
		if (!gStartSet || !gRange || !gDP)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG	
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	printf("********************************************************************************\r\n");
	printf("*                   AMDKangaroo v1.0  AMD ROCm/HIP Port                        *\r\n");
	printf("*          Original: (c) 2024 RetiredCoder  |  AMD Port: (c) 2025 Sirius437   *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("AMD ROCm/HIP port of RCKangaroo - https://github.com/RetiredC\r\n");
	printf("Fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
	printf("DEBUG MODE\r\n\r\n");
#endif

	InitEc();
	gDP = 0;
	gRange = 0;
	gStartSet = false;
	gTamesFileName[0] = 0;
	gMax = 0.0;
	gGenMode = false;
	gIsOpsLimit = false;
	gKernelBenchmark = false;
	gBenchmarkOptionSeen = false;
	gBenchmarkConfig.WarmupIterations = 3;
	gBenchmarkConfig.TimedIterations = 9;
	gBenchmarkConfig.Seed = 0x5EED950942ULL;
	gBenchmarkOutput[0] = 0;
	memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
	if (!ParseCommandLine(argc, argv))
		return 0;

	InitGpus();

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

	if (gKernelBenchmark)
	{
		printf("\r\nDETERMINISTIC KERNEL BENCHMARK MODE\r\n\r\n");
		const bool benchmarkSucceeded = RunKernelBenchmark();
		for (int index = 0; index < GpuCnt; index++)
			delete GpuKangs[index];
		DeInitEc();
		return benchmarkSucceeded ? 0 : 1;
	}

	pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	TotalOps = 0;
	TotalSolved = 0;
	gTotalErrors = 0;
	IsBench = gPubKey.x.IsZero();

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk, pk_found;

		PntToSolve = gPubKey;
		if (!gStart.IsZero())
		{
			PntOfs = ec.MultiplyG(gStart);
			PntOfs.y.NegModP();
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
		}

		char sx[100], sy[100];
		gPubKey.x.GetHexStr(sx);
		gPubKey.y.GetHexStr(sy);
		printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
		gStart.GetHexStr(sx);
		printf("Offset: %s\r\n", sx);

		if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
		{
			if (!gIsOpsLimit)
				printf("FATAL ERROR: SolvePoint failed\r\n");
			goto label_end;
		}
		pk_found.AddModP(gStart);
		EcPoint tmp = ec.MultiplyG(pk_found);
		if (!tmp.IsEqual(gPubKey))
		{
			printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
			goto label_end;
		}
		//happy end
		char s[100];
		pk_found.GetHexStr(s);
		printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);
		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else //we cannot save the key, show error and wait forever so the key is displayed
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");
		//solve points, show K
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;

			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 16;

			//generate random pk
			pk.RndBits(gRange);
			PntToSolve = ec.MultiplyG(pk);

			if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
			{
				if (!gIsOpsLimit)
					printf("FATAL ERROR: SolvePoint failed\r\n");
				break;
			}
			if (!pk_found.IsEqual(pk))
			{
				printf("FATAL ERROR: Found key is wrong!\r\n");
				break;
			}
			TotalOps += PntTotalOps;
			TotalSolved++;
			u64 ops_per_pnt = TotalOps / TotalSolved;
			double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
			printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
			//if (TotalSolved >= 100) break; //dbg
		}
	}
label_end:
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
	free(pPntList2);
	free(pPntList);
}

