#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>

#include "../Ec.h"
#include "../GpuArch.h"

hipError_t CallGpuKernelGen(TKparams params, const char*& failedOperation);
hipError_t CallGpuKernelABForTest(TKparams params, const char*& failedOperation);
hipError_t cuSetGpuParams(u64* jumpTable);

namespace
{
constexpr u32 TestBlockCount = 1;
constexpr u32 TestKangarooCount = TestBlockCount * BLOCK_SIZE * PNT_GROUP_CNT;
constexpr u64 InitialDistanceBase = 1ULL << 48;

static_assert(STEP_CNT == 1,
	"The kernel-state test must compile both host and device code with STEP_CNT=1");

struct JumpReference
{
	EcInt Distance;
	EcPoint Point;
	EcPoint NegativePoint;
};

struct StateSample
{
	u32 KangarooIndex;
	u32 HistoryGroup;
	u32 HistoryLane;
	EcInt Distance;
	EcPoint Point;
};

void FreeTestAllocation(void* pointer, const char* name);

struct DeviceBuffers
{
	u64* Kangs = nullptr;
	u64* L2 = nullptr;
	u32* DPsOut = nullptr;
	u64* Jumps1 = nullptr;
	u64* Jumps2 = nullptr;
	u64* JumpsList = nullptr;
	u32* DPTable = nullptr;
	u32* L1S2 = nullptr;
	u64* LastPoints = nullptr;
	u64* LoopTable = nullptr;
	u32* Debug = nullptr;
	u32* LoopedKangaroos = nullptr;

	~DeviceBuffers()
	{
		// Test cleanup must also work after a partial allocation failure.
		FreeTestAllocation(LoopedKangaroos, "hipFree(LoopedKangs)");
		FreeTestAllocation(Debug, "hipFree(dbg_buf)");
		FreeTestAllocation(LoopTable, "hipFree(LoopTable)");
		FreeTestAllocation(LastPoints, "hipFree(LastPnts)");
		FreeTestAllocation(L1S2, "hipFree(L1S2)");
		FreeTestAllocation(DPTable, "hipFree(DPTable)");
		FreeTestAllocation(JumpsList, "hipFree(JumpsList)");
		FreeTestAllocation(Jumps2, "hipFree(Jumps2)");
		FreeTestAllocation(Jumps1, "hipFree(Jumps1)");
		FreeTestAllocation(DPsOut, "hipFree(DPs_out)");
		FreeTestAllocation(L2, "hipFree(L2)");
		FreeTestAllocation(Kangs, "hipFree(Kangs)");
	}
};

bool CheckHip(hipError_t status, const char* operation)
{
	if (status == hipSuccess)
		return true;
	std::fprintf(stderr, "%s failed: %s (HIP error %d)\n", operation,
		hipGetErrorString(status), static_cast<int>(status));
	return false;
}

void FreeTestAllocation(void* pointer, const char* name)
{
	if (pointer)
		CheckHip(hipFree(pointer), name);
}

template <typename T>
bool Allocate(T*& pointer, size_t bytes, const char* name)
{
	const hipError_t status =
		hipMalloc(reinterpret_cast<void**>(&pointer), bytes);
	if (status == hipSuccess)
		return true;
	std::fprintf(stderr, "hipMalloc(%s, %zu bytes) failed: %s (HIP error %d)\n",
		name, bytes, hipGetErrorString(status), static_cast<int>(status));
	return false;
}

bool ParseDeviceIndex(int argc, char** argv, int& deviceIndex)
{
	deviceIndex = 0;
	if (argc == 1)
		return true;
	if (argc != 3 || std::strcmp(argv[1], "--gpu") != 0)
		return false;

	errno = 0;
	char* end = nullptr;
	const long parsed = std::strtol(argv[2], &end, 10);
	if (errno || !end || *end != '\0' || parsed < 0 || parsed > 31)
		return false;
	deviceIndex = static_cast<int>(parsed);
	return true;
}

EcInt MakeInitialDistance(u32 kangarooIndex)
{
	// Every kangaroo starts far enough above the small jump distances that the
	// test never crosses zero even when it takes 100 consecutive negative jumps.
	EcInt distance;
	distance.Set(InitialDistanceBase + static_cast<u64>(kangarooIndex + 1) * 0x10001ULL);
	return distance;
}

std::vector<JumpReference> BuildJumpTable(
	std::vector<u64>& fullTable, std::vector<u64>& compactPointTable)
{
	std::vector<JumpReference> jumps(JMP_CNT);
	fullTable.assign(static_cast<size_t>(JMP_CNT) * 12, 0);
	compactPointTable.assign(static_cast<size_t>(JMP_CNT) * 8, 0);

	for (u32 index = 0; index < JMP_CNT; index++)
	{
		// Even distances match the solver's production jump-table invariant.
		jumps[index].Distance.Set(0x1000ULL + 2ULL * index);
		jumps[index].Point = Ec::MultiplyG(jumps[index].Distance);
		jumps[index].NegativePoint = jumps[index].Point;
		jumps[index].NegativePoint.y.NegModP();

		u64* full = fullTable.data() + static_cast<size_t>(index) * 12;
		std::memcpy(full, jumps[index].Point.x.data, 32);
		std::memcpy(full + 4, jumps[index].Point.y.data, 32);
		std::memcpy(full + 8, jumps[index].Distance.data, 32);

		u64* compact = compactPointTable.data() + static_cast<size_t>(index) * 8;
		std::memcpy(compact, jumps[index].Point.x.data, 32);
		std::memcpy(compact + 4, jumps[index].Point.y.data, 32);
	}
	return jumps;
}

u32 KernelBKangarooIndex(u32 pairIndex, u32 lane, bool secondInPair)
{
	// KernelB consumes KernelA's wave-interleaved stream in pairs. Reproducing
	// this mapping lets the test inspect histories from wave and group edges.
	const u32 combinedIndex = lane + pairIndex * BLOCK_SIZE;
	const u32 waveIndex = combinedIndex / (32 * PNT_GROUP_CNT / 2);
	const u32 laneWithinWave = (combinedIndex / 4) % 32;
	const u32 groupOfEight = (combinedIndex % (32 * PNT_GROUP_CNT / 2)) / 128;
	const u32 groupWithinEight = 2 * (combinedIndex % 4);
	const u32 first = (32 * waveIndex + laneWithinWave) * PNT_GROUP_CNT
		+ 8 * groupOfEight + groupWithinEight;
	return first + (secondInPair ? 1 : 0);
}

void AddSample(std::vector<StateSample>& samples, u32 pairIndex, u32 lane,
	bool secondInPair)
{
	StateSample sample{};
	sample.KangarooIndex =
		KernelBKangarooIndex(pairIndex, lane, secondInPair);
	sample.HistoryGroup = 2 * pairIndex + (secondInPair ? 1 : 0);
	sample.HistoryLane = lane;
	sample.Distance = MakeInitialDistance(sample.KangarooIndex);
	sample.Point = Ec::MultiplyG(sample.Distance);

	const auto duplicate = std::find_if(samples.begin(), samples.end(),
		[&sample](const StateSample& existing)
		{
			return existing.KangarooIndex == sample.KangarooIndex;
		});
	if (duplicate == samples.end())
		samples.push_back(sample);
}

std::vector<StateSample> BuildSamples()
{
	std::vector<StateSample> samples;
	const std::array<u32, 9> boundaryLanes = {0, 1, 31, 32, 63, 64, 127, 128, 255};
	for (size_t index = 0; index < boundaryLanes.size(); index++)
		if (boundaryLanes[index] < BLOCK_SIZE)
			AddSample(samples, 0, boundaryLanes[index], (index & 1) != 0);

	// Derive middle and final pairs from the compiled group count. Fixed pair
	// numbers would read beyond smaller tuning profiles and hide real failures
	// behind invalid test samples.
	const u32 pairCount = PNT_GROUP_CNT / 2;
	const std::array<u32, 2> boundaryPairs = {
		pairCount / 2, pairCount - 1
	};
	const std::array<u32, 4> representativeLanes = {
		0, std::min<u32>(31, BLOCK_SIZE - 1),
		std::min<u32>(128, BLOCK_SIZE - 1), BLOCK_SIZE - 1
	};
	for (u32 pair : boundaryPairs)
		for (size_t index = 0; index < representativeLanes.size(); index++)
			AddSample(samples, pair, representativeLanes[index],
				(index & 1) != 0);
	return samples;
}

std::vector<u64> BuildInitialState()
{
	// Kangs is a 12-plane SoA. X/Y begin at zero because KernelGen derives the
	// point from the three distance limbs; plane 11 is reserved padding.
	std::vector<u64> state(static_cast<size_t>(TestKangarooCount) * 12, 0);
	for (u32 index = 0; index < TestKangarooCount; index++)
	{
		const EcInt distance = MakeInitialDistance(index);
		state[index + 8ULL * TestKangarooCount] = distance.data[0];
		state[index + 9ULL * TestKangarooCount] = distance.data[1];
		state[index + 10ULL * TestKangarooCount] = distance.data[2];
	}
	return state;
}

void AdvanceReference(StateSample& sample,
	std::vector<JumpReference>& jumps)
{
	const u32 jumpIndex = sample.Point.x.data[0] & JMP_MASK;
	const bool inverse = (sample.Point.y.data[0] & 1) != 0;
	EcPoint jumpPoint =
		inverse ? jumps[jumpIndex].NegativePoint : jumps[jumpIndex].Point;
	sample.Point = Ec::AddPoints(sample.Point, jumpPoint);
	if (inverse)
		sample.Distance.Sub(jumps[jumpIndex].Distance);
	else
		sample.Distance.Add(jumps[jumpIndex].Distance);
}

bool CompareLimbs(const char* field, u32 kangarooIndex, u32 step,
	const u64* actual, const u64* expected, size_t limbCount)
{
	for (size_t limb = 0; limb < limbCount; limb++)
	{
		if (actual[limb] == expected[limb])
			continue;
		std::fprintf(stderr,
			"Step %u, kangaroo %u: %s limb %zu mismatch "
			"(expected 0x%016llX, actual 0x%016llX)\n",
			step, kangarooIndex, field, limb,
			static_cast<unsigned long long>(expected[limb]),
			static_cast<unsigned long long>(actual[limb]));
		return false;
	}
	return true;
}

bool ValidateSamples(const std::vector<u64>& state,
	const std::vector<StateSample>& samples, u32 step)
{
	bool valid = true;
	for (const StateSample& sample : samples)
	{
		u64 actualX[4];
		u64 actualY[4];
		u64 actualDistance[3];
		for (u32 limb = 0; limb < 4; limb++)
		{
			actualX[limb] =
				state[sample.KangarooIndex + static_cast<u64>(limb) * TestKangarooCount];
			actualY[limb] =
				state[sample.KangarooIndex + static_cast<u64>(limb + 4) * TestKangarooCount];
		}
		for (u32 limb = 0; limb < 3; limb++)
			actualDistance[limb] =
				state[sample.KangarooIndex + static_cast<u64>(limb + 8) * TestKangarooCount];

		valid &= CompareLimbs("X", sample.KangarooIndex, step,
			actualX, sample.Point.x.data, 4);
		valid &= CompareLimbs("Y", sample.KangarooIndex, step,
			actualY, sample.Point.y.data, 4);
		valid &= CompareLimbs("distance", sample.KangarooIndex, step,
			actualDistance, sample.Distance.data, 3);
	}
	return valid;
}

bool ValidateFirstStepHistories(const std::vector<u64>& histories,
	const std::vector<StateSample>& samples)
{
	bool valid = true;
	for (const StateSample& sample : samples)
	{
		// After one jump, KernelB rotates history slot zero into physical slot
		// MD_LEN-1. Every selected lane must contain its own updated distance.
		const size_t offset =
			(static_cast<size_t>(sample.HistoryGroup) * MD_LEN + (MD_LEN - 1))
			* BLOCK_SIZE + sample.HistoryLane;
		const u64 actual = histories[offset];
		const u64 expected = sample.Distance.data[0];
		if (actual == expected)
			continue;
		std::fprintf(stderr,
			"History group %u lane %u mismatch "
			"(expected 0x%016llX, actual 0x%016llX)\n",
			sample.HistoryGroup, sample.HistoryLane,
			static_cast<unsigned long long>(expected),
			static_cast<unsigned long long>(actual));
		valid = false;
	}
	return valid;
}

bool AllocateBuffers(DeviceBuffers& buffers, TKparams& params)
{
	const size_t kangarooBytes =
		static_cast<size_t>(TestKangarooCount) * 12 * sizeof(u64);
	const size_t dpTableWords = static_cast<size_t>(TestKangarooCount)
		+ static_cast<size_t>(TestKangarooCount) * DPTABLE_MAX_CNT * 4;

	if (!Allocate(buffers.Kangs, kangarooBytes, "Kangs") ||
		!Allocate(buffers.L2, kangarooBytes, "L2") ||
		!Allocate(buffers.DPsOut, MAX_DP_CNT * GPU_DP_SIZE + 16, "DPs_out") ||
		!Allocate(buffers.Jumps1, JMP_CNT * 12 * sizeof(u64), "Jumps1") ||
		!Allocate(buffers.Jumps2, JMP_CNT * 12 * sizeof(u64), "Jumps2") ||
		!Allocate(buffers.JumpsList,
			static_cast<size_t>(TestKangarooCount) * STEP_CNT * sizeof(u16),
			"JumpsList") ||
		!Allocate(buffers.DPTable, dpTableWords * sizeof(u32), "DPTable") ||
		!Allocate(buffers.L1S2, BLOCK_SIZE * sizeof(u64), "L1S2") ||
		!Allocate(buffers.LastPoints,
			static_cast<size_t>(TestKangarooCount) * MD_LEN * 8 * sizeof(u64),
			"LastPnts") ||
		!Allocate(buffers.LoopTable,
			static_cast<size_t>(TestKangarooCount) * MD_LEN * sizeof(u64),
			"LoopTable") ||
		!Allocate(buffers.Debug, 256 * sizeof(u32), "dbg_buf") ||
		!Allocate(buffers.LoopedKangaroos,
			(TestKangarooCount + 2) * sizeof(u32), "LoopedKangs"))
		return false;

	params.Kangs = buffers.Kangs;
	params.KangCnt = TestKangarooCount;
	params.KangStride = TestKangarooCount;
	params.BlockCnt = TestBlockCount;
	params.BlockSize = BLOCK_SIZE;
	params.GroupCnt = PNT_GROUP_CNT;
	params.L2 = buffers.L2;
	params.DP = 60;
	params.DPs_out = buffers.DPsOut;
	params.Jumps1 = buffers.Jumps1;
	params.Jumps2 = buffers.Jumps2;
	params.JumpsList = buffers.JumpsList;
	params.DPTable = buffers.DPTable;
	params.L1S2 = buffers.L1S2;
	params.LastPnts = buffers.LastPoints;
	params.LoopTable = buffers.LoopTable;
	params.dbg_buf = buffers.Debug;
	params.LoopedKangs = buffers.LoopedKangaroos;
	params.IsGenMode = true;
	params.KernelA_LDS_Size = 64 * JMP_CNT + 16 * BLOCK_SIZE;
	params.KernelB_LDS_Size = 64 * JMP_CNT;
	return true;
}

bool InitializeDeviceState(const DeviceBuffers& buffers, TKparams& params,
	const std::vector<u64>& initialState, const std::vector<u64>& fullJumpTable,
	const std::vector<u64>& compactPointTable)
{
	const size_t stateBytes = initialState.size() * sizeof(u64);
	const size_t dpTableBytes = (static_cast<size_t>(TestKangarooCount)
		+ static_cast<size_t>(TestKangarooCount) * DPTABLE_MAX_CNT * 4)
		* sizeof(u32);
	if (!CheckHip(hipMemcpy(buffers.Kangs, initialState.data(), stateBytes,
			hipMemcpyHostToDevice), "copy initial kangaroo state") ||
		!CheckHip(hipMemcpy(buffers.Jumps1, fullJumpTable.data(),
			fullJumpTable.size() * sizeof(u64), hipMemcpyHostToDevice),
			"copy jump table 1") ||
		!CheckHip(hipMemcpy(buffers.Jumps2, fullJumpTable.data(),
			fullJumpTable.size() * sizeof(u64), hipMemcpyHostToDevice),
			"copy jump table 2") ||
		!CheckHip(cuSetGpuParams(const_cast<u64*>(compactPointTable.data())),
			"copy constant jump table 2") ||
		!CheckHip(hipMemset(buffers.DPsOut, 0, MAX_DP_CNT * GPU_DP_SIZE + 16),
			"clear distinguished-point output") ||
		!CheckHip(hipMemset(buffers.DPTable, 0, dpTableBytes),
			"clear distinguished-point table") ||
		!CheckHip(hipMemset(buffers.L1S2, 0, BLOCK_SIZE * sizeof(u64)),
			"clear level-one loop state") ||
		!CheckHip(hipMemset(buffers.LoopTable, 0,
			static_cast<size_t>(TestKangarooCount) * MD_LEN * sizeof(u64)),
			"clear loop history") ||
		!CheckHip(hipMemset(buffers.Debug, 0, 256 * sizeof(u32)),
			"clear debug counters") ||
		!CheckHip(hipMemset(buffers.LoopedKangaroos, 0,
			(TestKangarooCount + 2) * sizeof(u32)),
			"clear looped-kangaroo queue"))
		return false;

	const char* failedOperation = nullptr;
	return CheckHip(CallGpuKernelGen(params, failedOperation),
		failedOperation ? failedOperation : "KernelGen");
}

int RunStateTest()
{
	std::vector<u64> fullJumpTable;
	std::vector<u64> compactPointTable;
	std::vector<JumpReference> jumps =
		BuildJumpTable(fullJumpTable, compactPointTable);
	std::vector<StateSample> samples = BuildSamples();
	const std::vector<u64> initialState = BuildInitialState();

	DeviceBuffers buffers;
	TKparams params{};
	if (!AllocateBuffers(buffers, params) ||
		!InitializeDeviceState(buffers, params, initialState,
			fullJumpTable, compactPointTable))
		return 2;

	std::vector<u64> state(initialState.size());
	if (!CheckHip(hipMemcpy(state.data(), buffers.Kangs,
			state.size() * sizeof(u64), hipMemcpyDeviceToHost),
			"copy KernelGen state to host") ||
		!ValidateSamples(state, samples, 0))
		return 1;

	const std::array<u32, 4> checkpoints = {1, 2, 10, 100};
	for (u32 step = 1; step <= checkpoints.back(); step++)
	{
		for (StateSample& sample : samples)
			AdvanceReference(sample, jumps);

		const char* failedOperation = nullptr;
		if (!CheckHip(CallGpuKernelABForTest(params, failedOperation),
				failedOperation ? failedOperation : "one-step KernelA/KernelB"))
			return 2;

		if (step == 1)
		{
			std::vector<u64> histories(
				static_cast<size_t>(TestKangarooCount) * MD_LEN);
			if (!CheckHip(hipMemcpy(histories.data(), buffers.LoopTable,
					histories.size() * sizeof(u64), hipMemcpyDeviceToHost),
					"copy first-step loop histories") ||
				!ValidateFirstStepHistories(histories, samples))
				return 1;
		}

		if (std::find(checkpoints.begin(), checkpoints.end(), step)
			== checkpoints.end())
			continue;
		if (!CheckHip(hipMemcpy(state.data(), buffers.Kangs,
				state.size() * sizeof(u64), hipMemcpyDeviceToHost),
				"copy checkpoint kangaroo state") ||
			!ValidateSamples(state, samples, step))
			return 1;
		std::printf("  checkpoint %u: %zu sampled kangaroos matched\n",
			step, samples.size());
	}

	std::printf("PASS: KernelGen and one-step production KernelA/KernelB state "
		"matched the CPU curve reference at 1, 2, 10, and 100 jumps\n");
	return 0;
}
} // namespace

int main(int argc, char** argv)
{
	int deviceIndex = 0;
	if (!ParseDeviceIndex(argc, argv, deviceIndex))
	{
		std::fprintf(stderr, "Usage: %s [--gpu INDEX]\n", argv[0]);
		return 2;
	}

	int deviceCount = 0;
	if (!CheckHip(hipGetDeviceCount(&deviceCount), "hipGetDeviceCount"))
		return 2;
	if (deviceIndex >= deviceCount)
	{
		std::fprintf(stderr, "GPU %d is unavailable; detected %d device(s)\n",
			deviceIndex, deviceCount);
		return 2;
	}

	hipDeviceProp_t properties{};
	if (!CheckHip(hipGetDeviceProperties(&properties, deviceIndex),
			"hipGetDeviceProperties"))
		return 2;
	const AMDGpuProfile* profile = FindAMDGpuProfile(properties.gcnArchName);
	if (!profile || std::strcmp(profile->Name, GetCompiledGpuArchitecture()) != 0)
	{
		std::fprintf(stderr,
			"GPU %d architecture %s cannot run this %s state-test binary\n",
			deviceIndex, properties.gcnArchName, GetCompiledGpuArchitecture());
		return 2;
	}
	if (!CheckHip(hipSetDevice(deviceIndex), "hipSetDevice for kernel-state test"))
		return 2;

	std::printf("GPU %d: %s (%s), %u kangaroos, one production jump per launch\n",
		deviceIndex, properties.name, profile->Name, TestKangarooCount);
	InitEc();
	const int result = RunStateTest();
	DeInitEc();
	return result;
}
