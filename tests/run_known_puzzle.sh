#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
	echo "Usage: $0 SOLVER [GPU_INDEX] [TIMEOUT_SECONDS]" >&2
	exit 2
fi

solver=$(realpath "$1")
gpu_index=${2:-0}
timeout_seconds=${3:-60}

case "$gpu_index" in
	''|*[!0-9]*)
		echo "GPU_INDEX must be a non-negative integer" >&2
		exit 2
		;;
esac

case "$timeout_seconds" in
	''|*[!0-9]*|0)
		echo "TIMEOUT_SECONDS must be a positive integer" >&2
		exit 2
		;;
esac

if [ ! -x "$solver" ]; then
	echo "Solver is not executable: $solver" >&2
	exit 2
fi

# Run outside the source tree so a successful solve writes RESULTS.TXT only
# into an isolated temporary directory. The trap removes it on every exit path.
work_directory=$(mktemp -d)
trap 'rm -rf -- "$work_directory"' EXIT HUP INT TERM
output_file="$work_directory/solver-output.log"

expected_key="00000000000000000000000000000000000000000000000000000001A96CA8D8"
if (
	cd "$work_directory"
	set -- "$solver"
	set -- "$@" -gpu "$gpu_index"
	set -- "$@" -dp 16 -range 32
	set -- "$@" -start 100000000
	set -- "$@" -pubkey 03a355aa5e2e09dd44bb46a4722e9336e9e3ee4ee4e7b7a0cf5785b283bf2ab579
	timeout "${timeout_seconds}s" "$@"
) >"$output_file" 2>&1; then
	:
else
	status=$?
	cat "$output_file"
	if [ "$status" -eq 124 ]; then
		echo "FAIL: known puzzle timed out after ${timeout_seconds} seconds" >&2
	else
		echo "FAIL: solver exited with status $status" >&2
	fi
	exit 1
fi

cat "$output_file"
if ! grep -Fq "PRIVATE KEY: $expected_key" "$output_file"; then
	echo "FAIL: solver output did not contain the expected private key" >&2
	exit 1
fi

echo "PASS: known 32-bit puzzle produced the expected private key"
