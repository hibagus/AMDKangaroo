// Reproducible GPU benchmark reporting and optional ROCm SMI telemetry.

#include "GpuBenchmarkReport.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <iomanip>
#include <sstream>
#include <string>

namespace
{
const char* CsvHeader =
	"timestamp_utc,record_type,sample_index,gpu_index,gpu_name,gpu_uuid,"
	"architecture,driver_version,runtime_version,wave_size,scheduler_units,"
	"physical_cus,nominal_core_clock_mhz,nominal_memory_clock_mhz,"
	"block_size,point_groups,range_bits,dp_bits,step_count,seed,"
	"warmup_iterations,timed_iterations,timed_duration_s,jumps_per_iteration,"
	"clear_ms,kernel_a_ms,kernel_b_ms,kernel_c_ms,kernel_total_ms,"
	"transfer_ms,total_iteration_ms,raw_gjumps_s,effective_gjumps_s,"
	"distinguished_points,looped_kangaroos,"
	"before_junction_temp_c,before_memory_temp_c,before_power_w,"
	"before_core_clock_mhz,before_memory_clock_mhz,"
	"after_junction_temp_c,after_memory_temp_c,after_power_w,"
	"after_core_clock_mhz,after_memory_clock_mhz";

bool IsCardName(const char* name)
{
	if (std::strncmp(name, "card", 4) != 0 || !name[4])
		return false;
	for (const char* character = name + 4; *character; character++)
		if (!std::isdigit(static_cast<unsigned char>(*character)))
			return false;
	return true;
}

int FindDrmCardIndex(const hipDeviceProp_t& properties)
{
	DIR* devices = opendir("/sys/bus/pci/devices");
	if (!devices)
		return -1;

	char pciPrefix[32];
	std::snprintf(pciPrefix, sizeof(pciPrefix), "%04x:%02x:%02x.",
		properties.pciDomainID, properties.pciBusID, properties.pciDeviceID);
	int cardIndex = -1;
	while (dirent* device = readdir(devices))
	{
		if (std::strncmp(device->d_name, pciPrefix, std::strlen(pciPrefix)) != 0)
			continue;

		const std::string drmPath =
			std::string("/sys/bus/pci/devices/") + device->d_name + "/drm";
		DIR* drm = opendir(drmPath.c_str());
		if (!drm)
			continue;
		while (dirent* entry = readdir(drm))
			if (IsCardName(entry->d_name))
			{
				cardIndex = std::atoi(entry->d_name + 4);
				break;
			}
		closedir(drm);
		if (cardIndex >= 0)
			break;
	}
	closedir(devices);
	return cardIndex;
}

bool ExtractJsonNumber(const std::string& json, const char* key, double& value)
{
	std::string quotedKey(1, '"');
	quotedKey += key;
	quotedKey += '"';
	size_t position = json.find(quotedKey);
	if (position == std::string::npos)
		return false;
	position = json.find(':', position + quotedKey.size());
	if (position == std::string::npos)
		return false;

	const size_t valueEnd = json.find_first_of(",}", position + 1);
	const std::string field =
		json.substr(position + 1, valueEnd == std::string::npos
			? std::string::npos : valueEnd - position - 1);
	const char* begin = field.c_str();
	while (*begin && !std::isdigit(static_cast<unsigned char>(*begin))
		&& *begin != '-' && *begin != '.')
		begin++;
	if (!*begin)
		return false;

	errno = 0;
	char* end = nullptr;
	const double parsed = std::strtod(begin, &end);
	if (errno || end == begin)
		return false;
	value = parsed;
	return true;
}

std::string CsvEscape(const std::string& value)
{
	std::string escaped(1, '"');
	for (char character : value)
	{
		if (character == '"')
			escaped += '"';
		escaped += character;
	}
	escaped += '"';
	return escaped;
}

std::string FormatVersion(int version)
{
	std::ostringstream output;
	output << version / 10000000 << '.'
		<< (version / 100000) % 100 << '.'
		<< version % 100000;
	return output.str();
}

std::string OptionalNumber(bool available, double value)
{
	if (!available)
		return {};
	std::ostringstream output;
	output << std::fixed << std::setprecision(3) << value;
	return output.str();
}

class CsvRow
{
public:
	void Add(const std::string& value)
	{
		if (!First)
			Output << ',';
		First = false;
		Output << value;
	}

	template <typename Value>
	void AddNumber(Value value)
	{
		std::ostringstream number;
		number << value;
		Add(number.str());
	}

	std::string String() const
	{
		return Output.str();
	}

private:
	bool First = true;
	std::ostringstream Output;
};

std::string BuildCsvRow(const char* recordType, int sampleIndex,
	const TGpuBenchmarkSample& sample,
	const TGpuBenchmarkMetadata& metadata,
	const TGpuBenchmarkConfig& config, const TGpuBenchmarkResult& result)
{
	const double kernelTotal = sample.Kernels.KernelA_Milliseconds
		+ sample.Kernels.KernelB_Milliseconds
		+ sample.Kernels.KernelC_Milliseconds;
	const double rawRate = sample.Kernels.KernelA_Milliseconds > 0.0
		? result.JumpsPerIteration
			/ (sample.Kernels.KernelA_Milliseconds * 1000000.0)
		: 0.0;
	const double effectiveRate = sample.TotalMilliseconds > 0.0
		? result.JumpsPerIteration / (sample.TotalMilliseconds * 1000000.0)
		: 0.0;

	CsvRow row;
	row.Add(CsvEscape(metadata.TimestampUtc));
	row.Add(CsvEscape(recordType));
	if (sampleIndex >= 0)
		row.AddNumber(sampleIndex);
	else
		row.Add({});
	row.AddNumber(metadata.DeviceIndex);
	row.Add(CsvEscape(metadata.DeviceName));
	row.Add(CsvEscape(metadata.DeviceUuid));
	row.Add(CsvEscape(metadata.Architecture));
	row.Add(CsvEscape(FormatVersion(metadata.DriverVersion)));
	row.Add(CsvEscape(FormatVersion(metadata.RuntimeVersion)));
	row.AddNumber(metadata.WaveSize);
	row.AddNumber(metadata.SchedulerUnits);
	row.AddNumber(metadata.PhysicalComputeUnits);
	row.AddNumber(metadata.NominalCoreClockKHz / 1000.0);
	row.AddNumber(metadata.NominalMemoryClockKHz / 1000.0);
	row.AddNumber(metadata.BlockSize);
	row.AddNumber(metadata.PointGroupCount);
	row.AddNumber(metadata.Range);
	row.AddNumber(metadata.DistinguishedPointBits);
	row.AddNumber(metadata.StepCount);
	row.AddNumber(config.Seed);
	row.AddNumber(config.WarmupIterations);
	row.AddNumber(config.TimedIterations);
	row.AddNumber(result.TimedDurationSeconds);
	row.AddNumber(result.JumpsPerIteration);
	row.AddNumber(sample.ClearMilliseconds);
	row.AddNumber(sample.Kernels.KernelA_Milliseconds);
	row.AddNumber(sample.Kernels.KernelB_Milliseconds);
	row.AddNumber(sample.Kernels.KernelC_Milliseconds);
	row.AddNumber(kernelTotal);
	row.AddNumber(sample.TransferMilliseconds);
	row.AddNumber(sample.TotalMilliseconds);
	row.AddNumber(rawRate);
	row.AddNumber(effectiveRate);
	row.AddNumber(sample.DistinguishedPointCount);
	row.AddNumber(sample.LoopedKangarooCount);
	row.Add(OptionalNumber(metadata.Before.Available,
		metadata.Before.JunctionTemperatureC));
	row.Add(OptionalNumber(metadata.Before.Available,
		metadata.Before.MemoryTemperatureC));
	row.Add(OptionalNumber(metadata.Before.Available,
		metadata.Before.SocketPowerW));
	row.Add(OptionalNumber(metadata.Before.Available,
		metadata.Before.CoreClockMHz));
	row.Add(OptionalNumber(metadata.Before.Available,
		metadata.Before.MemoryClockMHz));
	row.Add(OptionalNumber(metadata.After.Available,
		metadata.After.JunctionTemperatureC));
	row.Add(OptionalNumber(metadata.After.Available,
		metadata.After.MemoryTemperatureC));
	row.Add(OptionalNumber(metadata.After.Available,
		metadata.After.SocketPowerW));
	row.Add(OptionalNumber(metadata.After.Available,
		metadata.After.CoreClockMHz));
	row.Add(OptionalNumber(metadata.After.Available,
		metadata.After.MemoryClockMHz));
	return row.String();
}
} // namespace

std::string CurrentUtcTimestamp()
{
	const std::time_t now = std::time(nullptr);
	std::tm utc{};
	gmtime_r(&now, &utc);
	char buffer[32];
	std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
	return buffer;
}

std::string FormatGpuUuid(const hipUUID& uuid)
{
	std::ostringstream output;
	output << std::hex << std::setfill('0');
	for (size_t index = 0; index < sizeof(uuid.bytes); index++)
	{
		if (index == 4 || index == 6 || index == 8 || index == 10)
			output << '-';
		output << std::setw(2)
			<< static_cast<unsigned int>(
				static_cast<unsigned char>(uuid.bytes[index]));
	}
	return output.str();
}

bool CollectGpuTelemetry(const hipDeviceProp_t& properties,
	TGpuTelemetry& telemetry, std::string& warning)
{
	telemetry = {};
	const int cardIndex = FindDrmCardIndex(properties);
	if (cardIndex < 0)
	{
		warning = "could not map the HIP device to a DRM card";
		return false;
	}

	char command[192];
	std::snprintf(command, sizeof(command),
		"rocm-smi -d %d --showtemp --showpower --showclocks --json 2>/dev/null",
		cardIndex);
	FILE* pipe = popen(command, "r");
	if (!pipe)
	{
		warning = "could not start rocm-smi";
		return false;
	}

	std::string json;
	char buffer[4096];
	while (std::fgets(buffer, sizeof(buffer), pipe))
		json += buffer;
	const int status = pclose(pipe);
	if (status != 0)
	{
		warning = "rocm-smi returned a nonzero status";
		return false;
	}

	const bool complete =
		ExtractJsonNumber(json, "Temperature (Sensor junction) (C)",
			telemetry.JunctionTemperatureC)
		&& ExtractJsonNumber(json, "Temperature (Sensor memory) (C)",
			telemetry.MemoryTemperatureC)
		&& ExtractJsonNumber(json,
			"Current Socket Graphics Package Power (W)",
			telemetry.SocketPowerW)
		&& ExtractJsonNumber(json, "sclk clock speed:",
			telemetry.CoreClockMHz)
		&& ExtractJsonNumber(json, "mclk clock speed:",
			telemetry.MemoryClockMHz);
	if (!complete)
	{
		warning = "rocm-smi output did not contain every requested sensor";
		return false;
	}
	telemetry.Available = true;
	warning.clear();
	return true;
}

void PrintGpuBenchmarkSummary(const TGpuBenchmarkMetadata& metadata,
	const TGpuBenchmarkConfig& config, const TGpuBenchmarkResult& result)
{
	const TGpuBenchmarkSample& median = result.Median;
	const double kernelTotal = median.Kernels.KernelA_Milliseconds
		+ median.Kernels.KernelB_Milliseconds
		+ median.Kernels.KernelC_Milliseconds;
	const double rawRate = median.Kernels.KernelA_Milliseconds > 0.0
		? result.JumpsPerIteration
			/ (median.Kernels.KernelA_Milliseconds * 1000000.0)
		: 0.0;
	const double effectiveRate = median.TotalMilliseconds > 0.0
		? result.JumpsPerIteration
			/ (median.TotalMilliseconds * 1000000.0)
		: 0.0;

	std::printf(
		"Benchmark median (%u warm-up, %u timed, %.3f timed seconds):\n"
		"  KernelA: %.3f ms, KernelB: %.3f ms, KernelC: %.3f ms\n"
		"  Kernels: %.3f ms, clear: %.3f ms, transfers: %.3f ms, total: %.3f ms\n"
		"  Raw KernelA rate: %.3f GJumps/s, effective iteration rate: %.3f GJumps/s\n",
		config.WarmupIterations, config.TimedIterations,
		result.TimedDurationSeconds,
		median.Kernels.KernelA_Milliseconds,
		median.Kernels.KernelB_Milliseconds,
		median.Kernels.KernelC_Milliseconds,
		kernelTotal, median.ClearMilliseconds, median.TransferMilliseconds,
		median.TotalMilliseconds, rawRate, effectiveRate);
	if (metadata.After.Available)
		std::printf(
			"  Post-run telemetry: %.1f C junction, %.1f C memory, %.1f W, "
			"%.0f MHz core, %.0f MHz memory\n",
			metadata.After.JunctionTemperatureC,
			metadata.After.MemoryTemperatureC,
			metadata.After.SocketPowerW, metadata.After.CoreClockMHz,
			metadata.After.MemoryClockMHz);
}

bool AppendGpuBenchmarkCsv(const char* path,
	const TGpuBenchmarkMetadata& metadata, const TGpuBenchmarkConfig& config,
	const TGpuBenchmarkResult& result)
{
	FILE* output = std::fopen(path, "a+");
	if (!output)
	{
		std::fprintf(stderr, "Cannot open benchmark CSV '%s': %s\n",
			path, std::strerror(errno));
		return false;
	}

	std::fseek(output, 0, SEEK_END);
	const long size = std::ftell(output);
	if (size == 0)
		std::fprintf(output, "%s\n", CsvHeader);
	for (size_t index = 0; index < result.Samples.size(); index++)
	{
		const std::string row = BuildCsvRow("sample", static_cast<int>(index),
			result.Samples[index], metadata, config, result);
		std::fprintf(output, "%s\n", row.c_str());
	}
	const std::string summary =
		BuildCsvRow("median", -1, result.Median, metadata, config, result);
	std::fprintf(output, "%s\n", summary.c_str());

	const bool success = std::fclose(output) == 0;
	if (!success)
		std::fprintf(stderr, "Failed to close benchmark CSV '%s'\n", path);
	return success;
}
