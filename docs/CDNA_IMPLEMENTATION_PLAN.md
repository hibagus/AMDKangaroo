# CDNA Implementation and Optimization Plan

This is the durable implementation plan for bringing AMDKangaroo to AMD
Instinct MI300X (`gfx942`) and MI355X (`gfx950`). Update it whenever
measurements change the plan or a milestone is completed.

## Project decisions

- Keep one shared main branch instead of permanent per-GPU branches.
- Keep correctness and algorithm code shared between architectures.
- Use explicit compile-time architecture profiles for performance-sensitive
  values such as block size, point-group count, grid multiplier, and LDS use.
- Produce separate `gfx942` and `gfx950` optimized binaries. A fat binary can be
  offered for convenience, but it must not replace faster specialized builds.
- Make one focused commit per logical change. Each commit must build and pass
  the tests available for its scope.
- Add clear comments around non-obvious arithmetic, memory layouts, indexing,
  synchronization, and architecture-specific choices. Explain the reason and
  invariant rather than restating the syntax.
- Whenever behavior, build instructions, supported hardware, tuning defaults,
  results, or limitations change, update `README.md` in the same logical
  commit. A phase is not complete until its README documentation is accurate.

## Current baseline

The source now has architecture-specific builds and runtime profiles for
`gfx1100`, `gfx942`, and `gfx950`, but the CDNA implementation is not yet
correctness- or performance-validated.

Completed bring-up work:

- Build outputs and object directories are isolated by target architecture.
- Runtime selection uses HIP's `gcnArchName` and reports wave size, processor
  count, and LDS capacity without assuming a full physical device.
- Host launch geometry comes from the same 256-thread, 24-group constants used
  to compile the HIP kernels.
- MI355X completed a bounded kernel-launch smoke test without a HIP launch error.
- Fatal HIP API and kernel errors report the failed operation, stop the affected
  worker after its first error, and end the solve cleanly when no workers remain.
- AMD modular multiplication and squaring now fold final 256-bit carries,
  canonicalize results below the secp256k1 prime, and give the square
  accumulator a valid carry limb.
- A deterministic GPU harness validates 1,409 negate, add, subtract, multiply,
  square, and inverse cases against an independent host reference. All cases
  pass on MI355X; the MI300X test binary is compile-validated.

Remaining issues that must be resolved before performance can be trusted:

- `KernelB` uses the block index for four per-thread loop-history accesses.
- Forced LLVM inlining and unrolling make `KernelA` use 512 VGPRs and 948 bytes
  of scratch per thread on both targets. Plain `-O3` produces 202 VGPRs with no
  scratch for the same source.
- There are no automated kernel-state, end-to-end, or benchmark
  regression tests.

The available development system has ROCm 7.2 and eight MI355X GPUs. `gfx942`
code generation can be checked locally, but final MI300X performance and runtime
validation require access to an MI300X system.

## Phase 1: Correct CDNA bring-up

1. Parameterize the build for `gfx942` and `gfx950`.
2. Remove harmful forced compiler heuristics and retain compiler resource
   reports for comparison.
3. Detect the architecture from HIP's `gcnArchName`. Use `warpSize`,
   `multiProcessorCount`, and reported LDS capacity instead of CUDA heuristics.
4. Replace `IsOldGpu` with an explicit architecture profile.
5. Define launch geometry in one source of truth and verify that host launches
   agree with the kernel's compiled constants.
6. Correct CU reporting and support partitioned devices without assuming the
   full physical GPU is visible.
7. Propagate failures from allocation, initialization, launch, synchronization,
   and transfer paths, stopping a worker after its first fatal error.
8. Correct the `KernelB` loop-history indexing and audit translated indexing
   against the upstream CUDA implementation.
9. Document every state array and packed field, including ownership,
   dimensions, strides, alignment, and producer/consumer kernels.

### Phase 1 completion gate

- Both targets compile without warnings introduced by our code.
- MI355X completes a known small puzzle without a HIP error.
- MI300X completes the same test when hardware is available.
- Architecture, wave size, CU count, LDS, and selected profile are reported
  correctly.
- Unsupported architectures fail with a clear diagnostic.

## Phase 2: Correctness and regression tests

Add deterministic tests before changing field arithmetic or layouts:

- Modular negate, add, subtract, multiply, square, and inverse.
- Zero, one, `p - 1`, carry-heavy, borrow-heavy, and randomized operands.
- Point addition, point doubling, and starting-point generation.
- GPU state after 1, 2, 10, and 100 jumps compared with the CPU reference.
- Distinguished-point construction and output decoding.
- Loop detection, history maintenance, and loop escape.
- Known puzzle vectors from the README.
- Canary regions around GPU allocations during test builds.
- Deterministic seeds so failures can always be reproduced.

The portable arithmetic path remains the reference until every optimized
variant passes the same vectors.

### Phase 2 completion gate

- Tests pass on `gfx950` and on `gfx942` hardware when available.
- Results are identical across repeated runs with the same seed.
- No out-of-bounds canary is modified.
- Correctness failures prevent performance results from being accepted.

## Phase 3: Reproducible performance measurement

Add a benchmark mode with:

- Explicit GPU selection, fixed seed, warm-up count, timed iteration count, and
  workload duration.
- HIP event timings for kernels A, B, C, transfers, and total iteration time.
- Machine-readable JSON or CSV output.
- GPU UUID, architecture, ROCm version, tuning profile, DP value, step count,
  clocks, power, and temperature in each result.
- Jump-rate accounting that excludes initialization and distinguishes raw jumps
  per second from effective solver performance.
- A script that records VGPR, SGPR, LDS, scratch, and code size from generated
  ISA for every kernel.

Use `rocprofv3` and ROCm Compute Profiler to measure occupancy, active waves,
VALU utilization and stalls, instruction mix, cache and HBM traffic, LDS
traffic, atomics, and dispatch overhead.

### Benchmark acceptance rules

- Compare identical seeds, DP settings, puzzle ranges, and correctness checks.
- Use warm runs and report the median of repeated samples.
- Record thermal and clock state so throttled runs are rejected.
- Do not accept an optimization based only on compiler resource counts.
- Require an improvement larger than normal run-to-run noise.
- Report both architectures for shared changes. Architecture-specific changes
  must not silently regress the other architecture.

## Phase 4: Remove known overhead

1. Keep `KernelA` free of scratch spills.
2. Remove relocatable device code if measurement shows it is unnecessary.
3. Remove device-wide synchronization where stream ordering already expresses
   the dependency.
4. Disable debug-buffer and loop-counter transfers in release builds.
5. Let `KernelC` return uniformly before staging its jump table when there is no
   loop work.
6. Replace atomics where each kangaroo has a provably exclusive owner.
7. Aggregate global output reservations per wave when profiling shows
   contention.

## Phase 5: Architecture-specific launch tuning

Sweep rather than assume the best values:

- Block size: 128, 256, and 512 where legal.
- Point-group count: 8, 12, 16, 24, and 32.
- Grid size: 1 through 4 workgroups per visible CU.
- Step count: 250, 500, 1000, and 2000.
- Launch bounds and compiler register limits.

`gfx942` has 64 KiB LDS per CU. Reducing KernelA below 32 KiB may allow two
resident workgroups. `gfx950` has 160 KiB LDS per CU and should test multiple
resident workgroups as well as larger, performance-positive LDS caches.

Every literal lane count must be documented as either an actual wave-size
dependency or an intentional packing width. Wave64 conversion must not be
performed mechanically.

## Phase 6: Field-arithmetic optimization

Create isolated microbenchmarks for `MulModP`, `SqrModP`, and `InvModP`, then:

1. Compare 8-by-32-bit and 4-by-64-bit limb representations.
2. Implement and test a secp256k1-specific pseudo-Mersenne reduction.
3. Compare Clang carry/multiply builtins with minimal AMD GCN instructions.
4. Reduce the square implementation's large temporary working set.
5. Compare fixed-iteration inversion with the current data-dependent method.
6. Reject variants that introduce spills or excessive register pressure.

Handwritten ISA is acceptable only when it has a measured advantage, retains a
portable reference implementation, and passes every arithmetic test.

## Phase 7: LDS, cache, and global-memory optimization

- Compare full jump tables in LDS with read-only cache access.
- Test storing only jump X coordinates in LDS and loading Y during the backward
  pass.
- Rework kangaroo state into a verified coalesced, group-major SoA layout.
- Replace opaque indexing macros with typed and aligned helpers when generated
  code remains equal or better.
- Align hot allocations to the relevant cache-line size.
- Measure L1, L2, Infinity Cache, and HBM traffic before redesigning the packed
  jump-history buffer.
- Investigate sparse or compressed DP and loop-history storage.

## Phase 8: Host and multi-GPU scaling

- Use dedicated streams and pinned, double-buffered output memory.
- Overlap host distinguished-point processing with the next GPU iteration.
- Test HIP graphs only after measuring dispatch overhead.
- Bind GPU worker threads to appropriate CPU/PCIe NUMA nodes.
- Batch or shard host-table insertion if it becomes contended.
- Measure 1-, 2-, 4-, and 8-GPU scaling.
- Identify devices by UUID and handle compute partitions as independent agents.

## Planned commit sequence

The sequence can be refined by measurements, but each entry remains a separate
logical commit:

1. [x] `build: add gfx942 and gfx950 target profiles`
2. [x] `runtime: detect CDNA devices from HIP architecture properties`
3. [x] `runtime: fail fast on HIP API and kernel errors`
4. [x] `fix: canonicalize AMD multiplication and square reductions`
5. [x] `test: add deterministic GPU field arithmetic validation`
6. `fix: repair kernel history indexing and document layouts`
7. `test: validate kangaroo state and known puzzle solutions`
8. `benchmark: add reproducible per-kernel performance reporting`
9. `perf: remove spill-inducing compiler overrides`
10. `perf(gfx942): tune launch geometry and LDS occupancy`
11. `perf(gfx950): tune launch geometry and LDS occupancy`
12. `perf: optimize secp256k1 field arithmetic`
13. `perf: reduce jump-table and loop-processing overhead`
14. `perf: overlap transfers and host DP processing`
15. `perf: improve multi-GPU host scaling`
16. `docs: publish validated MI300X and MI355X results`

Tests and directly related README updates belong in the same logical commit as
the behavior they validate or document. Independent test infrastructure or
documentation restructuring remains its own commit.

## Final completion criteria

- All correctness suites pass on both target architectures.
- Both targets pass an extended soak test without HIP errors, state corruption,
  output overflow, or lost workers.
- Kernel resource and profiler measurements are archived with final results.
- Single- and multi-GPU results are reproducible.
- Architecture defaults are selected automatically and remain overridable.
- The README accurately describes supported GPUs, builds, usage, validated
  performance, limitations, and benchmark methodology.
- No planned phase remains unchecked or undocumented.

## Progress checklist

- [ ] Phase 1: Correct CDNA bring-up
- [ ] Phase 2: Correctness and regression tests
- [ ] Phase 3: Reproducible performance measurement
- [ ] Phase 4: Remove known overhead
- [ ] Phase 5: Architecture-specific launch tuning
- [ ] Phase 6: Field-arithmetic optimization
- [ ] Phase 7: LDS, cache, and global-memory optimization
- [ ] Phase 8: Host and multi-GPU scaling
- [ ] Final README and validated-results review
