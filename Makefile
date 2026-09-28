CC := g++
HIPCC := hipcc
AS := as
ROCM_PATH ?= /opt/rocm

# Build GPU objects in architecture-specific directories. This prevents Make
# from reusing a code object compiled for a different GPU after GPU_ARCH changes.
SUPPORTED_GPU_ARCHS := gfx1100 gfx942 gfx950
GPU_ARCH ?= gfx950
BUILD_ROOT := build
BUILD_DIR := $(BUILD_ROOT)/$(GPU_ARCH)
# Keep the historical executable name for a default or GPU_ARCH-selected build.
# Explicit architecture targets below use names that can coexist.
TARGET ?= amdkangaroo

ifeq ($(filter $(GPU_ARCH),$(SUPPORTED_GPU_ARCHS)),)
$(error Unsupported GPU_ARCH '$(GPU_ARCH)'; choose one of: $(SUPPORTED_GPU_ARCHS))
endif

# Enable ASM primitives (comment out to disable)
USE_ASM_PRIMITIVES := 1

# Host compiler flags are shared because the CPU code is architecture-neutral.
CCFLAGS := -O3 -march=native -mtune=native -ffast-math -funroll-loops \
           -finline-functions -fomit-frame-pointer \
           -fno-stack-protector -fno-plt -fprefetch-loop-arrays \
           -ftree-vectorize \
           -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__

# Assembly flags for x86-64 with AVX2
# --64: Generate 64-bit code
# --noexecstack: Mark stack as non-executable (security)
# Note: GNU as doesn't support -march, CPU features are in the assembly code
ASFLAGS := --64 --noexecstack

ifdef USE_ASM_PRIMITIVES
CCFLAGS += -DUSE_ASM_PRIMITIVES
endif

# GPU_ARCH selects one code object per binary so CDNA targets can be tuned
# independently without weakening the existing gfx1100 build.
# The inherited LLVM overrides are intentionally retained in this build-only
# change. A later measured change will remove or tune them.
# -O3: Maximum optimization
# -fgpu-rdc: Relocatable device code for separate compilation
# -Rpass-analysis: Report kernel resource usage when supported by the compiler
GPU_ARCH_FLAG := --offload-arch=$(GPU_ARCH)
# Record the code-object target in host code for an early runtime compatibility check.
GPU_ARCH_CPPFLAG := -DAMDK_TARGET_ARCH=\"$(GPU_ARCH)\"
CCFLAGS += $(GPU_ARCH_CPPFLAG)
HIPCCFLAGS := -O3 $(GPU_ARCH_FLAG) $(GPU_ARCH_CPPFLAG) -fgpu-rdc -D__HIP_PLATFORM_AMD__ \
              -ffast-math -munsafe-fp-atomics \
              -mllvm -amdgpu-early-inline-all=true \
              -mllvm -unroll-threshold=1000 \
              -mllvm -inline-threshold=10000 \
              -Rpass-analysis=kernel-resource-usage

LDFLAGS := -L$(ROCM_PATH)/lib -lamdhip64 -pthread

CPU_SRC := AMDKangaroo.cpp GpuKang.cpp GpuArch.cpp Ec.cpp utils.cpp
GPU_SRC := AMDGpuCore.hip

# ASM primitives (only if enabled)
ifdef USE_ASM_PRIMITIVES
ASM_SRC := secp256k1_asm_full.s inverse256_skylake.s
CPU_SRC += InvModP_wrapper.cpp
else
ASM_SRC :=
endif

CPP_OBJECTS := $(addprefix $(BUILD_DIR)/,$(CPU_SRC:.cpp=.o))
HIP_OBJECTS := $(addprefix $(BUILD_DIR)/,$(GPU_SRC:.hip=.o))
ASM_OBJECTS := $(addprefix $(BUILD_DIR)/,$(ASM_SRC:.s=.o))

all: $(TARGET)

.PHONY: all all-cdna all-arch clean gfx1100 gfx942 gfx950

# These convenience targets use recursive Make invocations so each architecture
# receives its own variables and object directory, including under parallel Make.
gfx1100 gfx942 gfx950:
	$(MAKE) GPU_ARCH=$@ TARGET=amdkangaroo-$@ all

all-cdna: gfx942 gfx950

all-arch: gfx1100 all-cdna

$(TARGET): $(CPP_OBJECTS) $(HIP_OBJECTS) $(ASM_OBJECTS)
	$(HIPCC) $(GPU_ARCH_FLAG) -fgpu-rdc $(CCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.hip
	@mkdir -p $(@D)
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.s
	@mkdir -p $(@D)
	$(AS) $(ASFLAGS) $< -o $@

clean:
	$(RM) -r $(BUILD_ROOT)
	$(RM) amdkangaroo amdkangaroo-gfx1100 amdkangaroo-gfx942 amdkangaroo-gfx950
