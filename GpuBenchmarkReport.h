// Metadata and CSV serialization for reproducible GPU benchmark results.

#pragma once

#include <string>

#include <hip/hip_runtime.h>

#include "GpuBenchmark.h"

struct TGpuTelemetry
{
	bool Available = false;
	double JunctionTemperatureC = 0.0;
	double MemoryTemperatureC = 0.0;
	double SocketPowerW = 0.0;
	double CoreClockMHz = 0.0;
	double MemoryClockMHz = 0.0;
};

struct TGpuBenchmarkMetadata
{
	std::string TimestampUtc;
	int DeviceIndex = 0;
	std::string DeviceName;
	std::string DeviceUuid;
	std::string Architecture;
	int DriverVersion = 0;
	int RuntimeVersion = 0;
	int WaveSize = 0;
	int SchedulerUnits = 0;
	int PhysicalComputeUnits = 0;
	int NominalCoreClockKHz = 0;
	int NominalMemoryClockKHz = 0;
	u32 BlockSize = 0;
	u32 PointGroupCount = 0;
	u32 Range = 0;
	u32 DistinguishedPointBits = 0;
	u32 StepCount = 0;
	TGpuTelemetry Before;
	TGpuTelemetry After;
};

std::string CurrentUtcTimestamp();
std::string FormatGpuUuid(const hipUUID& uuid);

// Telemetry is optional because rocm-smi may be absent or restricted. A missing
// sample never invalidates kernel timings; the CSV records empty sensor fields.
bool CollectGpuTelemetry(const hipDeviceProp_t& properties,
	TGpuTelemetry& telemetry, std::string& warning);

void PrintGpuBenchmarkSummary(const TGpuBenchmarkMetadata& metadata,
	const TGpuBenchmarkConfig& config, const TGpuBenchmarkResult& result);

bool AppendGpuBenchmarkCsv(const char* path,
	const TGpuBenchmarkMetadata& metadata, const TGpuBenchmarkConfig& config,
	const TGpuBenchmarkResult& result);
