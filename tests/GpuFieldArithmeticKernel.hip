#include <hip/hip_runtime.h>

#include "GpuFieldArithmeticTest.h"
#include "../AMDGpuUtils.h"

namespace
{
__global__ void TestFieldArithmetic(const GpuFieldTestInput* inputs,
	GpuFieldTestOutput* outputs, size_t testCount)
{
	const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= testCount)
		return;

	// Exercise the in-place aliasing used by the solver, for example
	// MulModP(inverse, inverse, factor), instead of testing an easier copy.
	u64 value[5] = {};
	u64 right[4];
	for (int limb = 0; limb < 4; limb++)
	{
		value[limb] = inputs[index].Left[limb];
		right[limb] = inputs[index].Right[limb];
	}

	switch (static_cast<GpuFieldOperation>(inputs[index].Operation))
	{
	case GpuFieldOperation::Negate:
		NegModP(value);
		break;
	case GpuFieldOperation::Add:
		AddModP(value, value, right);
		break;
	case GpuFieldOperation::Subtract:
		SubModP(value, value, right);
		break;
	case GpuFieldOperation::Multiply:
		MulModP(value, value, right);
		break;
	case GpuFieldOperation::Square:
		SqrModP(value, value);
		break;
	case GpuFieldOperation::Inverse:
		InvModP(reinterpret_cast<u32*>(value));
		break;
	}

	for (int limb = 0; limb < 5; limb++)
		outputs[index].Value[limb] = value[limb];
}
} // namespace

hipError_t RunGpuFieldArithmeticTests(int deviceIndex, const GpuFieldTestInput* inputs,
	GpuFieldTestOutput* outputs, size_t testCount, const char*& failedOperation)
{
	GpuFieldTestInput* deviceInputs = nullptr;
	GpuFieldTestOutput* deviceOutputs = nullptr;
	hipError_t status = hipSuccess;
	constexpr unsigned int threadsPerBlock = 256;
	const unsigned int blockCount =
		static_cast<unsigned int>((testCount + threadsPerBlock - 1) / threadsPerBlock);

	failedOperation = "hipSetDevice for field arithmetic test";
	status = hipSetDevice(deviceIndex);
	if (status != hipSuccess)
		goto cleanup;

	failedOperation = "hipMalloc(field test inputs)";
	status = hipMalloc(reinterpret_cast<void**>(&deviceInputs),
		testCount * sizeof(GpuFieldTestInput));
	if (status != hipSuccess)
		goto cleanup;

	failedOperation = "hipMalloc(field test outputs)";
	status = hipMalloc(reinterpret_cast<void**>(&deviceOutputs),
		testCount * sizeof(GpuFieldTestOutput));
	if (status != hipSuccess)
		goto cleanup;

	failedOperation = "hipMemcpy(field test inputs to device)";
	status = hipMemcpy(deviceInputs, inputs, testCount * sizeof(GpuFieldTestInput),
		hipMemcpyHostToDevice);
	if (status != hipSuccess)
		goto cleanup;

	failedOperation = "field arithmetic test kernel launch";
	TestFieldArithmetic<<<blockCount, threadsPerBlock>>>(deviceInputs, deviceOutputs, testCount);
	status = hipGetLastError();
	if (status != hipSuccess)
		goto cleanup;

	failedOperation = "field arithmetic test kernel execution";
	status = hipDeviceSynchronize();
	if (status != hipSuccess)
		goto cleanup;

	failedOperation = "hipMemcpy(field test outputs to host)";
	status = hipMemcpy(outputs, deviceOutputs, testCount * sizeof(GpuFieldTestOutput),
		hipMemcpyDeviceToHost);

cleanup:
	// Preserve the first failure. Cleanup errors are useful only when all test
	// work completed successfully.
	if (deviceOutputs)
	{
		const hipError_t freeStatus = hipFree(deviceOutputs);
		if (status == hipSuccess && freeStatus != hipSuccess)
		{
			failedOperation = "hipFree(field test outputs)";
			status = freeStatus;
		}
	}
	if (deviceInputs)
	{
		const hipError_t freeStatus = hipFree(deviceInputs);
		if (status == hipSuccess && freeStatus != hipSuccess)
		{
			failedOperation = "hipFree(field test inputs)";
			status = freeStatus;
		}
	}

	if (status == hipSuccess)
		failedOperation = nullptr;
	return status;
}
