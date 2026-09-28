#pragma once

// Architectures with an explicitly supported build and runtime profile.
enum class AMDGpuArchitecture
{
	Gfx1100,
	Gfx942,
	Gfx950
};

struct AMDGpuProfile
{
	AMDGpuArchitecture Architecture;
	const char* Name;
	const char* ProcessorUnitName;
	int ComputeUnitsPerProcessor;
	int BlockSize;
	int PointGroupCount;
	bool UsesL2Workspace;
};

// HIP may append feature flags such as ":sramecc+:xnack-" to gcnArchName.
// This lookup accepts those flags while still matching the base architecture
// name exactly.
const AMDGpuProfile* FindAMDGpuProfile(const char* gcnArchName);

// The Makefile supplies this value so a single-architecture code object cannot
// silently be selected for a different GPU model at runtime.
const char* GetCompiledGpuArchitecture();
