#!/usr/bin/make -f
################################################################################
# AMDKangaroo - Fast GPU ECDLP Solver for AMD GPUs
#
# This Makefile builds AMDKangaroo optimized for:
#   - RDNA 3: AMD Radeon RX 7900 XTX (gfx1100)
#   - CDNA 3: AMD Mi300X (gfx942)
#   - CDNA 4: AMD Mi355X (gfx950)
#
# Usage:
#   make                           # Build for RDNA 3 (default)
#   make OFFLOAD_ARCH=gfx942      # Build for Mi300X (CDNA 3)
#   make OFFLOAD_ARCH=gfx950      # Build for Mi355X (CDNA 4)
#   make clean                     # Remove build artifacts
#
# ROCm Installation:
#   This Makefile auto-detects ROCm 10 at /opt/rocm/core-10
#   or falls back to /opt/rocm or system default
################################################################################

# ============================================================================
# Toolchain Configuration
# ============================================================================

CC := g++
HIPCC := hipcc
AS := as

# Auto-detect ROCm path: ROCm 10 (core-10) > system ROCm > /opt/rocm
ROCM_PATH ?= $(shell if [ -d /opt/rocm/core-10 ]; then echo /opt/rocm/core-10; \
                      elif [ -d /opt/rocm ]; then echo /opt/rocm; \
                      else echo /opt/rocm; fi)

$(info Building with ROCm at: $(ROCM_PATH))

# ============================================================================
# Build Configuration Flags
# ============================================================================

# GPU Architecture Selection
# Default: gfx1100 (RDNA 3 - RX 7900 XTX)
# Options:
#   gfx942 = Mi300X (CDNA 3, Wave64)
#   gfx950 = Mi355X (CDNA 4, Wave64)
OFFLOAD_ARCH ?= gfx1100

# Enable ASM primitives (10-20% performance boost)
# Disable by commenting out: # USE_ASM_PRIMITIVES := 1
USE_ASM_PRIMITIVES := 1

# ============================================================================
# CPU Compilation Flags
# ============================================================================
# These flags optimize CPU-side code (host code)
#
# Key optimizations:
#   -O3               : Aggressive optimization level
#   -march=native     : Optimize for local CPU architecture
#   -mtune=native     : Tune for local CPU
#   -ffast-math       : Fast but approximate math (safe for crypto bitwise ops)
#   -funroll-loops    : Unroll loops for better performance
#   -finline-functions: Inline functions for reduced call overhead
#   -fomit-frame-pointer: Save register by omitting frame pointers
#   -fno-stack-protector: Disable stack canary (not needed in compute code)
#   -ftree-vectorize  : Enable automatic vectorization
#   -fprefetch-loop-arrays: Prefetch data for loops

CCFLAGS := -O3 -march=native -mtune=native -ffast-math -funroll-loops \
           -finline-functions -fomit-frame-pointer \
           -fno-stack-protector -fno-plt -fprefetch-loop-arrays \
           -ftree-vectorize -std=c++11 \
           -I$(ROCM_PATH)/include \
           -I./include \
           -D__HIP_PLATFORM_AMD__

# ============================================================================
# GPU Compilation Flags
# ============================================================================
# These flags optimize GPU kernel compilation
#
# Key optimizations:
#   -O3                           : Maximum optimization
#   --offload-arch=gfx*           : Target specific GPU architecture
#   -fgpu-rdc                     : Relocatable device code (separate compilation)
#   -ffast-math                   : Fast math operations (safe for this algorithm)
#   -munsafe-fp-atomics          : Faster atomic operations (acceptable for ECDLP)
#   -mllvm -amdgpu-early-inline-all=true  : Aggressive inlining of device functions
#   -mllvm -unroll-threshold=1500 : Unroll loops with high threshold
#   -mllvm -inline-threshold=10000: Inline functions with high threshold
#   -mllvm -enable-load-store-vectorizer=true: Enable vector load/store optimization
#   -Rpass-analysis=kernel-resource-usage: Report kernel resource usage
#
# CDNA 3/4 specific optimizations:
#   - Higher unroll threshold (1500 vs 1000) exploits Wave64 deeper pipelines
#   - Better instruction scheduling for Wave64 synchronization patterns
#   - Vectorization for L2 cache efficiency

HIPCCFLAGS := -O3 \
              --offload-arch=$(OFFLOAD_ARCH) \
              -fgpu-rdc -D__HIP_PLATFORM_AMD__ \
              -ffast-math -munsafe-fp-atomics \
              -mllvm -amdgpu-early-inline-all=true \
              -mllvm -unroll-threshold=1500 \
              -mllvm -inline-threshold=10000 \
              -mllvm -amdgpu-load-store-vectorizer=true \
              -Rpass-analysis=kernel-resource-usage \
              -I$(ROCM_PATH)/include \
              -I./include

# ============================================================================
# Assembler Flags (for x86-64 ASM optimizations)
# ============================================================================
# GNU assembler flags for x86-64 assembly code
#   --64       : Generate 64-bit code (required for x86-64)
#   --noexecstack: Mark stack as non-executable (security hardening)

ASFLAGS := --64 --noexecstack

# ============================================================================
# Linker Configuration
# ============================================================================
# Link against ROCm runtime libraries
LDFLAGS := -L$(ROCM_PATH)/lib -lamdhip64 -pthread

# ============================================================================
# Source Files Organization
# ============================================================================
# Organize sources by category for clarity and maintainability

# GPU kernel sources (HIP)
GPU_SRC := src/gpu/AMDGpuCore.hip

# CPU source files (regular C++)
CPU_SRC := src/cpu/AMDKangaroo.cpp \
           src/cpu/GpuKang.cpp \
           src/cpu/Ec.cpp \
           src/cpu/utils.cpp

# Assembly sources (x86-64 optimizations)
ifdef USE_ASM_PRIMITIVES
ASM_SRC := src/asm/secp256k1_asm_full.s src/asm/inverse256_skylake.s
CPU_SRC += src/cpu/InvModP_wrapper.cpp
else
ASM_SRC :=
endif

# ============================================================================
# Build Artifacts
# ============================================================================
# Map source files to object files

CPU_OBJECTS := $(CPU_SRC:.cpp=.o)
HIP_OBJECTS := $(GPU_SRC:.hip=.o)
ASM_OBJECTS := $(ASM_SRC:.s=.o)

# Final executable name
TARGET := build/amdkangaroo

# ============================================================================
# Build Rules
# ============================================================================

.PHONY: all clean help info

all: info $(TARGET)

# Create build directory if needed
$(TARGET): build $(CPU_OBJECTS) $(HIP_OBJECTS) $(ASM_OBJECTS)
	@echo "Linking executable for $(OFFLOAD_ARCH)..."
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) -fgpu-rdc $(CCFLAGS) \
	         -o $@ $(CPU_OBJECTS) $(HIP_OBJECTS) $(ASM_OBJECTS) $(LDFLAGS)
	@echo "Build complete: $@"
	@echo "To run: ./$@ -dp 16 -range 32 -start 100000000 -pubkey <key>"

# Compile CPU code with include path
%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@

# Compile GPU kernels with include path
%.o: %.hip
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@

# Assemble x86-64 code
%.o: %.s
	@mkdir -p $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# Create build directory
build:
	@mkdir -p build

# ============================================================================
# Utility Targets
# ============================================================================

help:
	@echo "AMDKangaroo Makefile - GPU ECDLP Solver"
	@echo ""
	@echo "Targets:"
	@echo "  make                  - Build for RDNA 3 (RX 7900 XTX, gfx1100)"
	@echo "  make OFFLOAD_ARCH=gfx942 - Build for Mi300X (CDNA 3)"
	@echo "  make OFFLOAD_ARCH=gfx950 - Build for Mi355X (CDNA 4)"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make help             - Show this help message"
	@echo "  make info             - Show build configuration"
	@echo ""
	@echo "Configuration:"
	@echo "  ROCM_PATH    - Path to ROCm installation (default: auto-detect)"
	@echo "  OFFLOAD_ARCH - Target GPU architecture (default: gfx1100)"
	@echo "  USE_ASM_PRIMITIVES - Enable x86-64 ASM optimizations (default: 1)"
	@echo ""
	@echo "Example:"
	@echo "  make OFFLOAD_ARCH=gfx942 USE_ASM_PRIMITIVES=1"

info:
	@echo "=== AMDKangaroo Build Configuration ==="
	@echo "ROCm Path:       $(ROCM_PATH)"
	@echo "GPU Architecture: $(OFFLOAD_ARCH)"
	@echo "ASM Primitives:  $(if $(USE_ASM_PRIMITIVES),enabled,disabled)"
	@echo "Target:          $(TARGET)"
	@echo "====================================="

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(CPU_OBJECTS) $(HIP_OBJECTS) $(ASM_OBJECTS)
	rm -rf build/
	@echo "Clean complete"

# ============================================================================
# Phony targets (not actual files)
# ============================================================================

.PHONY: all clean help info

# ============================================================================
# End of Makefile
# ============================================================================
