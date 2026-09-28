// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "Ec.h"
#include "GpuBenchmark.h"
#include "GpuArch.h"

#define STATS_WND_SIZE	16

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class AMDGpuKang
{
private:
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	u32* DPs_out;
	TKparams Kparams;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntA;
	EcPoint PntB;

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];

	void GenerateRndDistances();
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif
public:
	int persistingL2CacheMaxSize;
	int DeviceIndex; // Index used by the HIP runtime.
	int ProcessorCount; // HIP reports CUs on CDNA and WGPs on RDNA.
	int KangCnt;
	bool Failed;
	const AMDGpuProfile* Profile;

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();
	bool Benchmark(const TGpuBenchmarkConfig& config, TGpuBenchmarkResult& result);

	u32 dbg[256];

	int GetStatsSpeed();
};
