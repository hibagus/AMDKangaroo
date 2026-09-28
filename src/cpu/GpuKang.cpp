// RCKangaroo - AMD ROCm/HIP Port
// Original: (c) 2024 RetiredCoder (RC) - https://github.com/RetiredC
// License: GPLv3, see "LICENSE.TXT" file
// AMD Port: (c) 2025 Sirius437


#include <iostream>
#include <hip/hip_runtime.h>

#include "GpuKang.h"

hipError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode

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
	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	return Kparams.BlockSize* Kparams.GroupCnt* Kparams.BlockCnt;
}

//executes in main thread
bool AMDGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
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
	err = hipSetDevice(CudaIndex);
	if (err != hipSuccess)
		return false;

	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
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
	if (!IsOldGpu)
	{
		//L2	
		int L2size = Kparams.KangCnt * (3 * 32);
		total_mem += L2size;
		err = hipMalloc((void**)&Kparams.L2, L2size);
		if (err != hipSuccess)
		{
			printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, hipGetErrorString(err));
			return false;
		}
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
			printf("GPU %d, hipStreamSetAttribute failed: %s\n", CudaIndex, hipGetErrorString(err));
			return false;
		}
		*/
	}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.DPs_out, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	size = KangCnt * 96;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.Kangs, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.JumpsList, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = hipMalloc((void**)&Kparams.DPTable, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.L1S2, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LastPnts, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LoopTable, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += 1024;
	err = hipMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}

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
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy Jumps1 failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}
	free(buf);
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
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy Jumps2 failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}
	free(buf);

	err = cuSetGpuParams(Kparams, jmp2_table);
	if (err != hipSuccess)
	{
		free(jmp2_table);
		printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, hipGetErrorString(err));
		return false;
	}
	free(jmp2_table);
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = hipMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy Jumps3 failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}
	free(buf);

	printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
	return true;
}

void AMDGpuKang::Release()
{
	free(RndPnts);
	free(DPs_out);
	hipFree(Kparams.LoopedKangs);
	hipFree(Kparams.dbg_buf);
	hipFree(Kparams.LoopTable);
	hipFree(Kparams.LastPnts);
	hipFree(Kparams.L1S2);
	hipFree(Kparams.DPTable);
	hipFree(Kparams.JumpsList);
	hipFree(Kparams.Jumps3);
	hipFree(Kparams.Jumps2);
	hipFree(Kparams.Jumps1);
	hipFree(Kparams.Kangs);
	hipFree(Kparams.DPs_out);
	if (!IsOldGpu)
		hipFree(Kparams.L2);
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
	err = hipSetDevice(CudaIndex);
	if (err != hipSuccess)
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
		printf("GPU %d, hipMemcpy failed: %s\n", CudaIndex, hipGetErrorString(err));
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
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy failed: %s\n", CudaIndex, hipGetErrorString(err));
		return false;
	}
	CallGpuKernelGen(Kparams);

	err = hipMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
	if (err != hipSuccess)
		return false;
	hipMemset(Kparams.dbg_buf, 0, 1024);
	hipMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	return true;
}

#ifdef DEBUG_MODE
int AMDGpuKang::Dbg_CheckKangs()
{
	int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
	u64* kangs = (u64*)malloc(kang_size);
	hipError_t err = hipMemcpy(kangs, Kparams.Kangs, kang_size, hipMemcpyDeviceToHost);
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

extern u32 gTotalErrors;

//executes in separate thread
void AMDGpuKang::Execute()
{
	hipSetDevice(CudaIndex);

	if (!Start())
	{
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	hipError_t err;	
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		hipMemset(Kparams.DPs_out, 0, 4);
		hipMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
		hipMemset(Kparams.LoopedKangs, 0, 8);
		CallGpuKernelABC(Kparams);
		int cnt;
		err = hipMemcpy(&cnt, Kparams.DPs_out, 4, hipMemcpyDeviceToHost);
		if (err != hipSuccess)
		{
			printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, hipGetErrorString(err));
			gTotalErrors++;
			break;
		}
		
		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = hipMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, hipMemcpyDeviceToHost);
			if (err != hipSuccess)
			{
				gTotalErrors++;
				break;
			}
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

		//dbg
		hipMemcpy(dbg, Kparams.dbg_buf, 1024, hipMemcpyDeviceToHost);

		u32 lcnt;
		hipMemcpy(&lcnt, Kparams.LoopedKangs, 4, hipMemcpyDeviceToHost);
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
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