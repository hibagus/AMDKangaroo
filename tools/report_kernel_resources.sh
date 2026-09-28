#!/bin/sh

set -eu

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 SOLVER GPU_ARCH" >&2
	exit 2
fi

solver=$(realpath "$1")
gpu_arch=$2
rocm_path=${ROCM_PATH:-/opt/rocm}
objcopy="$rocm_path/llvm/bin/llvm-objcopy"
bundler="$rocm_path/llvm/bin/clang-offload-bundler"
readelf="$rocm_path/llvm/bin/llvm-readelf"

if [ ! -x "$solver" ]; then
	echo "Solver is not executable: $solver" >&2
	exit 2
fi
for tool in "$objcopy" "$bundler" "$readelf"; do
	if [ ! -x "$tool" ]; then
		echo "Required ROCm tool is not executable: $tool" >&2
		exit 2
	fi
done

# Work only in a private temporary directory because the extracted code object
# and metadata are build artifacts, not source-controlled results.
work_directory=$(mktemp -d)
trap 'rm -rf -- "$work_directory"' EXIT HUP INT TERM
fat_binary="$work_directory/solver.fatbin"
code_object="$work_directory/kernel.co"
metadata="$work_directory/metadata.txt"
symbol_sizes="$work_directory/symbol-sizes.csv"

"$objcopy" --dump-section .hip_fatbin="$fat_binary" "$solver"
set -- "$bundler" --unbundle --type=o
set -- "$@" "--targets=hipv4-amdgcn-amd-amdhsa--$gpu_arch"
set -- "$@" "--input=$fat_binary" "--output=$code_object"
"$@"
"$readelf" --notes "$code_object" >"$metadata"

# Read the final linked symbols rather than LLVM's pre-link estimates.
"$readelf" --symbols --demangle "$code_object" |
	awk '
		$4 == "FUNC" && match($0, /Kernel(A|B|C|Gen)\(TKparams\)/) {
			name = substr($0, RSTART, RLENGTH)
			sub(/\(TKparams\)/, "", name)
			if (!(name in seen)) {
				print name "," $3
				seen[name] = 1
			}
		}
	' >"$symbol_sizes"

# Dynamic LDS is selected by the host at launch and is therefore absent from
# the code-object's fixed-LDS field. Derive it from the same compile-time
# constants used by the production host and device code.
block_size=$(g++ -E -dM -x c++ -D__HIP_PLATFORM_AMD__ defs.h |
	awk '$2 == "BLOCK_SIZE" { print $3 }')
jump_count=$(g++ -E -dM -x c++ -D__HIP_PLATFORM_AMD__ defs.h |
	awk '$2 == "JMP_CNT" { print $3 }')
kernel_a_lds=$((64 * jump_count + 16 * block_size))
kernel_b_lds=$((64 * jump_count))
kernel_c_lds=$((96 * jump_count))

echo "architecture,kernel,vgpr,agpr,sgpr,vgpr_spills,sgpr_spills,scratch_bytes_per_thread,fixed_lds_bytes,dynamic_lds_bytes,wave_size,code_size_bytes"
awk -v architecture="$gpu_arch" -v kernel_a_lds="$kernel_a_lds" -v kernel_b_lds="$kernel_b_lds" -v kernel_c_lds="$kernel_c_lds" '
	NR == FNR {
		split($0, symbol, ",")
		code_size[symbol[1]] = symbol[2]
		next
	}
	/- \.agpr_count:/ {
		agpr = $3
		fixed_lds = 0
		name = ""
		private_bytes = 0
		sgpr = 0
		sgpr_spills = 0
		vgpr = 0
		vgpr_spills = 0
		wave = 0
	}
	/\.group_segment_fixed_size:/ { fixed_lds = $2 }
	/\.name:/ {
		mangled_name = $2
		if (mangled_name ~ /KernelGen/)
			name = "KernelGen"
		else if (mangled_name ~ /KernelA/)
			name = "KernelA"
		else if (mangled_name ~ /KernelB/)
			name = "KernelB"
		else if (mangled_name ~ /KernelC/)
			name = "KernelC"
	}
	/\.private_segment_fixed_size:/ { private_bytes = $2 }
	/\.sgpr_count:/ { sgpr = $2 }
	/\.sgpr_spill_count:/ { sgpr_spills = $2 }
	/\.vgpr_count:/ { vgpr = $2 }
	/\.vgpr_spill_count:/ { vgpr_spills = $2 }
	/\.wavefront_size:/ {
		wave = $2
		if (name == "KernelA")
			dynamic_lds = kernel_a_lds
		else if (name == "KernelB")
			dynamic_lds = kernel_b_lds
		else if (name == "KernelC")
			dynamic_lds = kernel_c_lds
		else
			dynamic_lds = 0
		print architecture "," name "," vgpr "," agpr "," sgpr "," vgpr_spills "," sgpr_spills "," private_bytes "," fixed_lds "," dynamic_lds "," wave "," code_size[name]
	}
' "$symbol_sizes" "$metadata"
