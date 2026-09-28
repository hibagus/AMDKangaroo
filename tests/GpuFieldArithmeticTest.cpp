#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>
#include <hip/hip_runtime.h>

#include "GpuFieldArithmeticTest.h"
#include "../GpuArch.h"

namespace
{
using boost::multiprecision::cpp_int;

constexpr u64 DeterministicSeed = 0x5EED950942ULL;

struct FieldCase
{
	GpuFieldTestInput Input{};
	GpuFieldTestOutput Expected{};
};

const cpp_int& FieldPrime()
{
	// secp256k1 uses p = 2^256 - 2^32 - 977.
	static const cpp_int prime = (cpp_int(1) << 256) - (cpp_int(1) << 32) - 977;
	return prime;
}

void StoreLimbs(cpp_int value, u64* limbs, size_t limbCount)
{
	const cpp_int mask = (cpp_int(1) << 64) - 1;
	for (size_t limb = 0; limb < limbCount; limb++)
	{
		limbs[limb] = static_cast<u64>(value & mask);
		value >>= 64;
	}
}

cpp_int ModularInverse(cpp_int value)
{
	// Fermat's little theorem gives a^(p-2) mod p for every nonzero field value.
	cpp_int exponent = FieldPrime() - 2;
	cpp_int result = 1;
	while (exponent != 0)
	{
		if ((exponent & 1) != 0)
			result = (result * value) % FieldPrime();
		value = (value * value) % FieldPrime();
		exponent >>= 1;
	}
	return result;
}

FieldCase MakeCase(GpuFieldOperation operation, const cpp_int& left,
	const cpp_int& right = 0)
{
	FieldCase testCase;
	StoreLimbs(left, testCase.Input.Left, 4);
	StoreLimbs(right, testCase.Input.Right, 4);
	testCase.Input.Operation = static_cast<u32>(operation);

	cpp_int expected = 0;
	switch (operation)
	{
	case GpuFieldOperation::Negate:
		// The solver intentionally represents the negation of zero as p. Preserve
		// that exact limb encoding because later kernels consume these values.
		expected = left == 0 ? FieldPrime() : FieldPrime() - left;
		break;
	case GpuFieldOperation::Add:
		expected = (left + right) % FieldPrime();
		break;
	case GpuFieldOperation::Subtract:
		expected = left;
		expected -= right;
		if (expected < 0)
			expected += FieldPrime();
		break;
	case GpuFieldOperation::Multiply:
		expected = (left * right) % FieldPrime();
		break;
	case GpuFieldOperation::Square:
		expected = (left * left) % FieldPrime();
		break;
	case GpuFieldOperation::Inverse:
		expected = ModularInverse(left);
		break;
	}
	StoreLimbs(expected, testCase.Expected.Value, 5);
	return testCase;
}

void AddUnique(std::vector<cpp_int>& values, const cpp_int& value)
{
	if (std::find(values.begin(), values.end(), value) == values.end())
		values.push_back(value);
}

std::vector<FieldCase> BuildCases()
{
	const cpp_int& prime = FieldPrime();
	std::vector<cpp_int> edgeValues;
	AddUnique(edgeValues, 0);
	AddUnique(edgeValues, 1);
	AddUnique(edgeValues, 2);
	AddUnique(edgeValues, 3);
	AddUnique(edgeValues, (cpp_int(1) << 32) - 1);
	AddUnique(edgeValues, cpp_int(1) << 32);
	AddUnique(edgeValues, (cpp_int(1) << 64) - 1);
	AddUnique(edgeValues, cpp_int(1) << 64);
	AddUnique(edgeValues, (cpp_int(1) << 128) - 1);
	AddUnique(edgeValues, cpp_int(1) << 128);
	AddUnique(edgeValues, (cpp_int(1) << 192) - 1);
	AddUnique(edgeValues, cpp_int(1) << 192);
	AddUnique(edgeValues, cpp_int(1) << 255);
	AddUnique(edgeValues, prime / 2);
	AddUnique(edgeValues, prime - (cpp_int(1) << 32));
	AddUnique(edgeValues, prime - 2);
	AddUnique(edgeValues, prime - 1);
	AddUnique(edgeValues,
		cpp_int("0x0123456789ABCDEFFEDCBA987654321000000000FFFFFFFFA5A5A5A5A5A5A5A5"));

	std::vector<cpp_int> randomValues;
	std::mt19937_64 random(DeterministicSeed);
	for (int sample = 0; sample < 64; sample++)
	{
		std::array<u64, 4> randomLimbs;
		for (int limb = 0; limb < 4; limb++)
			randomLimbs[limb] = random();
		cpp_int value;
		boost::multiprecision::import_bits(value, randomLimbs.begin(),
			randomLimbs.end(), 64, false);
		AddUnique(randomValues, value % prime);
	}

	std::vector<FieldCase> cases;
	std::vector<cpp_int> unaryValues = edgeValues;
	unaryValues.insert(unaryValues.end(), randomValues.begin(), randomValues.end());
	for (const cpp_int& value : unaryValues)
	{
		cases.push_back(MakeCase(GpuFieldOperation::Negate, value));
		cases.push_back(MakeCase(GpuFieldOperation::Square, value));
		if (value != 0)
			cases.push_back(MakeCase(GpuFieldOperation::Inverse, value));
	}

	// The full edge-value cross product exercises carries and borrows at every
	// limb boundary, including both operand orders for subtraction.
	for (const cpp_int& left : edgeValues)
		for (const cpp_int& right : edgeValues)
		{
			cases.push_back(MakeCase(GpuFieldOperation::Add, left, right));
			cases.push_back(MakeCase(GpuFieldOperation::Subtract, left, right));
			cases.push_back(MakeCase(GpuFieldOperation::Multiply, left, right));
		}

	for (size_t sample = 0; sample < randomValues.size(); sample++)
	{
		const cpp_int& left = randomValues[sample];
		const cpp_int& right = randomValues[(sample * 37 + 11) % randomValues.size()];
		cases.push_back(MakeCase(GpuFieldOperation::Add, left, right));
		cases.push_back(MakeCase(GpuFieldOperation::Subtract, left, right));
		cases.push_back(MakeCase(GpuFieldOperation::Multiply, left, right));
	}
	return cases;
}

const char* OperationName(u32 operation)
{
	switch (static_cast<GpuFieldOperation>(operation))
	{
	case GpuFieldOperation::Negate: return "negate";
	case GpuFieldOperation::Add: return "add";
	case GpuFieldOperation::Subtract: return "subtract";
	case GpuFieldOperation::Multiply: return "multiply";
	case GpuFieldOperation::Square: return "square";
	case GpuFieldOperation::Inverse: return "inverse";
	}
	return "unknown";
}

std::string FormatValue(const u64* limbs, size_t limbCount)
{
	std::ostringstream stream;
	stream << "0x" << std::hex << std::uppercase;
	for (size_t limb = limbCount; limb-- > 0;)
		stream << std::setw(16) << std::setfill('0') << limbs[limb];
	return stream.str();
}

bool CheckHip(hipError_t status, const char* operation)
{
	if (status == hipSuccess)
		return true;
	std::fprintf(stderr, "%s failed: %s (HIP error %d)\n", operation,
		hipGetErrorString(status), static_cast<int>(status));
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
		std::fprintf(stderr, "GPU %d architecture %s cannot run this %s test binary\n",
			deviceIndex, properties.gcnArchName, GetCompiledGpuArchitecture());
		return 2;
	}

	std::vector<FieldCase> cases = BuildCases();
	std::vector<GpuFieldTestInput> inputs(cases.size());
	std::vector<GpuFieldTestOutput> outputs(cases.size());
	for (size_t index = 0; index < cases.size(); index++)
		inputs[index] = cases[index].Input;

	std::printf("GPU %d: %s (%s), %zu deterministic field-arithmetic cases, seed 0x%llX\n",
		deviceIndex, properties.name, profile->Name, cases.size(), DeterministicSeed);
	const char* failedOperation = nullptr;
	const hipError_t status = RunGpuFieldArithmeticTests(deviceIndex, inputs.data(),
		outputs.data(), cases.size(), failedOperation);
	if (!CheckHip(status, failedOperation ? failedOperation : "GPU field arithmetic test"))
		return 2;

	size_t failureCount = 0;
	std::array<size_t, 6> failuresByOperation{};
	for (size_t index = 0; index < cases.size(); index++)
	{
		if (std::memcmp(outputs[index].Value, cases[index].Expected.Value,
				sizeof(outputs[index].Value)) == 0)
			continue;

		const u32 operation = cases[index].Input.Operation;
		if (failuresByOperation[operation] < 3)
		{
			std::fprintf(stderr, "Case %zu (%s) failed\n  left:     %s\n",
				index, OperationName(cases[index].Input.Operation),
				FormatValue(cases[index].Input.Left, 4).c_str());
			if (cases[index].Input.Operation == static_cast<u32>(GpuFieldOperation::Add) ||
				cases[index].Input.Operation == static_cast<u32>(GpuFieldOperation::Subtract) ||
				cases[index].Input.Operation == static_cast<u32>(GpuFieldOperation::Multiply))
				std::fprintf(stderr, "  right:    %s\n",
					FormatValue(cases[index].Input.Right, 4).c_str());
			std::fprintf(stderr, "  expected: %s\n  actual:   %s\n",
				FormatValue(cases[index].Expected.Value, 5).c_str(),
				FormatValue(outputs[index].Value, 5).c_str());
		}
		failureCount++;
		failuresByOperation[operation]++;
	}

	if (failureCount)
	{
		std::fprintf(stderr, "FAIL: %zu of %zu field-arithmetic cases failed\n",
			failureCount, cases.size());
		for (u32 operation = 0; operation < failuresByOperation.size(); operation++)
			if (failuresByOperation[operation])
				std::fprintf(stderr, "  %s: %zu failure(s)\n",
					OperationName(operation), failuresByOperation[operation]);
		return 1;
	}

	std::printf("PASS: all %zu field-arithmetic cases matched the independent reference\n",
		cases.size());
	return 0;
}
