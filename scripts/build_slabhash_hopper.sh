#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/third_party/SlabHash"
patch_file="${repo_root}/patches/slabhash-sm90-cuda12.patch"

# The historical repository stores this file as CRLF; normalize it so the
# recorded patch applies identically on Linux and Windows checkouts.
sed -i 's/\r$//' "${source_dir}/CMakeLists.txt"

if git -C "${source_dir}" apply --check "${patch_file}"; then
  git -C "${source_dir}" apply "${patch_file}"
elif ! git -C "${source_dir}" apply --reverse --check "${patch_file}"; then
  echo "SlabHash Hopper patch is neither applicable nor already applied" >&2
  exit 1
fi

PATH=/usr/local/cuda/bin:${PATH} cmake -S "${source_dir}" -B "${source_dir}/build-hopper" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DSLABHASH_GENCODE_SM35=OFF \
  -DSLABHASH_GENCODE_SM90=ON \
  -DDGTEST=OFF
PATH=/usr/local/cuda/bin:${PATH} cmake --build "${source_dir}/build-hopper" -j
