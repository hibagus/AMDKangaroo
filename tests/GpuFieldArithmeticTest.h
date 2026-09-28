#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

#include "../defs.h"

// Keep this interface as plain fixed-width data so the host test and HIP kernel
// agree on every byte copied across PCIe.
enum class GpuFieldOperation : u32
{
	Negate,
	Add,
	Subtract,
	Multiply,
	Square,
	Inverse
};

struct GpuFieldTestInput
{
	u64 Left[4];
	u64 Right[4];
	u32 Operation;
	u32 Reserved; // Makes the structure's trailing padding explicit.
};

struct GpuFieldTestOutput
{
	// InvModP uses a ninth 32-bit scratch limb in place. Returning five limbs
	// lets the test detect an unexpected 256-bit overflow instead of hiding it.
	u64 Value[5];
};

static_assert(sizeof(GpuFieldTestInput) == 72,
	"Host and device field-test input layouts must remain identical");
static_assert(sizeof(GpuFieldTestOutput) == 40,
	"Host and device field-test output layouts must remain identical");

hipError_t RunGpuFieldArithmeticTests(int deviceIndex, const GpuFieldTestInput* inputs,
	GpuFieldTestOutput* outputs, std::size_t testCount, const char*& failedOperation);
