#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline_dir="${repo_root}/third_party/LPMA"

if [[ ! -f "${baseline_dir}/Makefile" ]]; then
  echo "LPMA submodule is missing; run: git submodule update --init" >&2
  exit 1
fi

# Keep the artifact source unmodified.  Its Makefile hard-codes sm_60, so
# override only the compiler command and retain separable compilation required
# by the GPMA dynamic-parallelism path.
nvcc_cmd="nvcc -arch=sm_90 -O3 -lineinfo -lcudadevrt -rdc=true"
make -C "${baseline_dir}" UPDATE_GPMA NVCC="${nvcc_cmd}"
make -C "${baseline_dir}" UPDATE_LPMA NVCC="${nvcc_cmd}"

echo "Built ${baseline_dir}/UPDATE_GPMA and UPDATE_LPMA"

