CC := g++
HIPCC := hipcc
AS := as
ROCM_PATH ?= /opt/rocm

# Build GPU objects in architecture-specific directories. This prevents Make
# from reusing a code object compiled for a different GPU after GPU_ARCH changes.
SUPPORTED_GPU_ARCHS := gfx1100 gfx942 gfx950
GPU_ARCH ?= gfx950

# MI355X measurements favor more independent state and the largest supported
# point batch. Keep MI300X and RDNA defaults conservative until measured on
# their actual hardware; every value remains explicitly overridable.
ifeq ($(GPU_ARCH),gfx950)
DEFAULT_GPU_BLOCK_SIZE := 256
DEFAULT_GPU_POINT_GROUP_COUNT := 32
DEFAULT_GPU_GRID_MULTIPLIER := 4
else
DEFAULT_GPU_BLOCK_SIZE := 256
DEFAULT_GPU_POINT_GROUP_COUNT := 24
DEFAULT_GPU_GRID_MULTIPLIER := 1
endif
GPU_BLOCK_SIZE ?= $(DEFAULT_GPU_BLOCK_SIZE)
GPU_POINT_GROUP_COUNT ?= $(DEFAULT_GPU_POINT_GROUP_COUNT)
GPU_GRID_MULTIPLIER ?= $(DEFAULT_GPU_GRID_MULTIPLIER)
BUILD_ROOT := build
# Include every compile-time tuning value in the object path. Benchmark sweeps
# can then switch configurations without accidentally linking stale objects.
TUNING_TAG := b$(GPU_BLOCK_SIZE)-g$(GPU_POINT_GROUP_COUNT)-x$(GPU_GRID_MULTIPLIER)
BUILD_DIR := $(BUILD_ROOT)/$(GPU_ARCH)/$(TUNING_TAG)
# Keep the historical executable name for a default or GPU_ARCH-selected build.
# Explicit architecture targets below use names that can coexist.
TARGET ?= amdkangaroo

ifeq ($(filter $(GPU_ARCH),$(SUPPORTED_GPU_ARCHS)),)
$(error Unsupported GPU_ARCH '$(GPU_ARCH)'; choose one of: $(SUPPORTED_GPU_ARCHS))
endif
ifeq ($(filter $(GPU_BLOCK_SIZE),128 256 512),)
$(error Unsupported GPU_BLOCK_SIZE '$(GPU_BLOCK_SIZE)'; choose one of: 128 256 512)
endif
# Jump descriptors are packed in groups of eight, so point-group counts must be
# multiples of eight until that layout is generalized.
ifeq ($(filter $(GPU_POINT_GROUP_COUNT),8 16 24 32),)
$(error Unsupported GPU_POINT_GROUP_COUNT '$(GPU_POINT_GROUP_COUNT)'; choose one of: 8 16 24 32)
endif
ifeq ($(filter $(GPU_GRID_MULTIPLIER),1 2 3 4),)
$(error Unsupported GPU_GRID_MULTIPLIER '$(GPU_GRID_MULTIPLIER)'; choose one of: 1 2 3 4)
endif

TUNING_CPPFLAGS := -DAMDK_BLOCK_SIZE=$(GPU_BLOCK_SIZE) \
                   -DAMDK_POINT_GROUP_COUNT=$(GPU_POINT_GROUP_COUNT) \
                   -DAMDK_GRID_MULTIPLIER=$(GPU_GRID_MULTIPLIER)

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
# -O3: Maximum optimization
# -fgpu-rdc: Relocatable device code for separate compilation
# Let the AMD backend choose inlining and unrolling. The inherited forced
# thresholds spill KernelA state to scratch on both CDNA targets.
# -Rpass-analysis: Report kernel resource usage when supported by the compiler
GPU_ARCH_FLAG := --offload-arch=$(GPU_ARCH)
# Record the code-object target in host code for an early runtime compatibility check.
GPU_ARCH_CPPFLAG := -DAMDK_TARGET_ARCH=\"$(GPU_ARCH)\"
CCFLAGS += $(GPU_ARCH_CPPFLAG) $(TUNING_CPPFLAGS)
HIPCCFLAGS := -O3 $(GPU_ARCH_FLAG) $(GPU_ARCH_CPPFLAG) $(TUNING_CPPFLAGS) -fgpu-rdc -D__HIP_PLATFORM_AMD__ \
              -ffast-math -munsafe-fp-atomics \
              -Rpass-analysis=kernel-resource-usage

LDFLAGS := -L$(ROCM_PATH)/lib -lamdhip64 -pthread

CPU_SRC := AMDKangaroo.cpp GpuKang.cpp GpuArch.cpp GpuBenchmarkReport.cpp \
           Ec.cpp utils.cpp
GPU_SRC := AMDGpuCore.hip

# The field test links the production device-arithmetic header into a small,
# architecture-specific kernel and compares its output with a host reference.
FIELD_TEST_TARGET ?= gpu-field-test-$(GPU_ARCH)
FIELD_TEST_OBJECTS := $(BUILD_DIR)/tests/GpuFieldArithmeticTest.o \
                      $(BUILD_DIR)/tests/GpuFieldArithmeticKernel.o $(BUILD_DIR)/GpuArch.o

STATE_TEST_TARGET ?= gpu-state-test-$(GPU_ARCH)

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

# The state test links the production kernels with STEP_CNT=1, then launches
# KernelA and KernelB repeatedly to inspect exact 1/2/10/100-jump checkpoints.
STATE_TEST_OBJECTS := $(BUILD_DIR)/tests/GpuKangarooStateTest.o \
                      $(BUILD_DIR)/tests/AMDGpuCoreStateTest.o \
                      $(BUILD_DIR)/GpuArch.o $(BUILD_DIR)/Ec.o \
                      $(BUILD_DIR)/utils.o $(ASM_OBJECTS)
ifdef USE_ASM_PRIMITIVES
STATE_TEST_OBJECTS += $(BUILD_DIR)/InvModP_wrapper.o
endif

all: $(TARGET)

.PHONY: all all-cdna all-arch clean gfx1100 gfx942 gfx950
.PHONY: field-test field-tests-cdna field-test-gfx942 field-test-gfx950
.PHONY: test-field test-field-gfx942 test-field-gfx950
.PHONY: state-test state-tests-cdna state-test-gfx942 state-test-gfx950
.PHONY: test-state test-state-gfx942 test-state-gfx950
.PHONY: test-known-puzzle test-known-puzzle-gfx942 test-known-puzzle-gfx950
.PHONY: kernel-resources kernel-resources-gfx942 kernel-resources-gfx950

# These convenience targets use recursive Make invocations so each architecture
# receives its own variables and object directory, including under parallel Make.
gfx1100 gfx942 gfx950:
	$(MAKE) GPU_ARCH=$@ TARGET=amdkangaroo-$@ all

all-cdna: gfx942 gfx950

all-arch: gfx1100 all-cdna

# Build both CDNA field-test binaries without trying to execute the gfx942
# binary on a development host that may only have gfx950 hardware.
field-tests-cdna: field-test-gfx942 field-test-gfx950

field-test-gfx942 field-test-gfx950:
	$(MAKE) GPU_ARCH=$(patsubst field-test-%,%,$@) \
		FIELD_TEST_TARGET=gpu-field-test-$(patsubst field-test-%,%,$@) field-test

# Run the test selected by GPU_ARCH. FIELD_TEST_GPU remains overridable for
# multi-GPU systems and defaults to the first visible device.
test-field: field-test
	./$(FIELD_TEST_TARGET) --gpu $(or $(FIELD_TEST_GPU),0)

test-field-gfx942 test-field-gfx950:
	$(MAKE) GPU_ARCH=$(patsubst test-field-%,%,$@) \
		FIELD_TEST_TARGET=gpu-field-test-$(patsubst test-field-%,%,$@) test-field

state-tests-cdna: state-test-gfx942 state-test-gfx950

state-test-gfx942 state-test-gfx950:
	$(MAKE) GPU_ARCH=$(patsubst state-test-%,%,$@) \
		STATE_TEST_TARGET=gpu-state-test-$(patsubst state-test-%,%,$@) state-test

test-state: state-test
	./$(STATE_TEST_TARGET) --gpu $(or $(STATE_TEST_GPU),0)

test-state-gfx942 test-state-gfx950:
	$(MAKE) GPU_ARCH=$(patsubst test-state-%,%,$@) \
		STATE_TEST_TARGET=gpu-state-test-$(patsubst test-state-%,%,$@) test-state

# The script runs in a temporary directory, checks the exact recovered key, and
# removes the generated RESULTS.TXT when it exits.
test-known-puzzle: $(TARGET)
	tests/run_known_puzzle.sh ./$(TARGET) $(or $(PUZZLE_TEST_GPU),0) \
		$(or $(PUZZLE_TEST_TIMEOUT),60)

test-known-puzzle-gfx942 test-known-puzzle-gfx950:
	$(MAKE) GPU_ARCH=$(patsubst test-known-puzzle-%,%,$@) \
		TARGET=amdkangaroo-$(patsubst test-known-puzzle-%,%,$@) test-known-puzzle

# Print final linked kernel resources as CSV. Unlike compiler remarks, this
# includes spill counts and code sizes from the executable's code object.
kernel-resources: $(TARGET)
	tools/report_kernel_resources.sh ./$(TARGET) $(GPU_ARCH) $(GPU_BLOCK_SIZE) \
		$(GPU_POINT_GROUP_COUNT) $(GPU_GRID_MULTIPLIER)

kernel-resources-gfx942 kernel-resources-gfx950:
	$(MAKE) GPU_ARCH=$(patsubst kernel-resources-%,%,$@) \
		TARGET=amdkangaroo-$(patsubst kernel-resources-%,%,$@) kernel-resources

$(TARGET): $(CPP_OBJECTS) $(HIP_OBJECTS) $(ASM_OBJECTS)
	$(HIPCC) $(GPU_ARCH_FLAG) -fgpu-rdc $(CCFLAGS) -o $@ $^ $(LDFLAGS)

field-test: $(FIELD_TEST_TARGET)

$(FIELD_TEST_TARGET): $(FIELD_TEST_OBJECTS)
	$(HIPCC) $(GPU_ARCH_FLAG) -fgpu-rdc $(CCFLAGS) -o $@ $^ $(LDFLAGS)

state-test: $(STATE_TEST_TARGET)

$(STATE_TEST_TARGET): $(STATE_TEST_OBJECTS)
	$(HIPCC) $(GPU_ARCH_FLAG) -fgpu-rdc $(CCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/tests/GpuKangarooStateTest.o: tests/GpuKangarooStateTest.cpp
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) -DSTEP_CNT=1 -c $< -o $@

$(BUILD_DIR)/tests/AMDGpuCoreStateTest.o: AMDGpuCore.hip
	@mkdir -p $(@D)
	$(HIPCC) $(HIPCCFLAGS) -DSTEP_CNT=1 -DAMDK_KERNEL_STATE_TEST \
		-c $< -o $@

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
	$(RM) gpu-field-test-gfx1100 gpu-field-test-gfx942 gpu-field-test-gfx950
	$(RM) gpu-state-test-gfx1100 gpu-state-test-gfx942 gpu-state-test-gfx950
