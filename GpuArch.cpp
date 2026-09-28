#include <cstring>

#include "defs.h"
#include "GpuArch.h"

#ifndef AMDK_TARGET_ARCH
#define AMDK_TARGET_ARCH "unknown"
#endif

namespace
{
// Each specialized binary receives its validated or experimental geometry from
// the Makefile. Runtime lookup keeps that geometry tied to the matching target.
const AMDGpuProfile SupportedProfiles[] = {
	{AMDGpuArchitecture::Gfx1100, "gfx1100", "WGPs", 2,
		BLOCK_SIZE, PNT_GROUP_CNT, AMDK_GRID_MULTIPLIER, true},
	{AMDGpuArchitecture::Gfx942, "gfx942", "CUs", 1,
		BLOCK_SIZE, PNT_GROUP_CNT, AMDK_GRID_MULTIPLIER, true},
	{AMDGpuArchitecture::Gfx950, "gfx950", "CUs", 1,
		BLOCK_SIZE, PNT_GROUP_CNT, AMDK_GRID_MULTIPLIER, true},
};

bool MatchesBaseArchitecture(const char* reportedName, const char* profileName)
{
	if (!reportedName || !profileName)
		return false;

	// gcnArchName normally starts with "gfx", but accepting a target triple makes
	// diagnostics and future unit tests robust to other ROCm name formats.
	const char* architecture = std::strstr(reportedName, "gfx");
	if (!architecture)
		return false;

	const size_t profileLength = std::strlen(profileName);
	if (std::strncmp(architecture, profileName, profileLength) != 0)
		return false;

	const char suffix = architecture[profileLength];
	return suffix == '\0' || suffix == ':';
}
} // namespace

const AMDGpuProfile* FindAMDGpuProfile(const char* gcnArchName)
{
	for (const AMDGpuProfile& profile : SupportedProfiles)
		if (MatchesBaseArchitecture(gcnArchName, profile.Name))
			return &profile;

	return nullptr;
}

const char* GetCompiledGpuArchitecture()
{
	return AMDK_TARGET_ARCH;
}
