#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${repo_root}/results"
build_dir="${BUILD_DIR:-${repo_root}/build}"
update_bin="${build_dir}/update_bench"
delta_bin="${build_dir}/lockfree_delta_bench"
publication_bin="${build_dir}/publication_bench"
nsys_bin="${NSYS_BIN:-/usr/local/bin/nsys}"
ncu_bin="${NCU_BIN:-/usr/local/cuda-12.9/bin/ncu}"
profile_gpu="${PROFILE_GPU:-0}"

mkdir -p "${result_dir}"

CUDA_VISIBLE_DEVICES="${profile_gpu}" "${nsys_bin}" profile \
  --trace=cuda,nvtx,osrt --sample=none --stats=true \
  --force-overwrite=true -o "${result_dir}/update_adaptive_nsys" \
  "${update_bin}" --vertices 4096 --segment-capacity 2048 \
  --density 0.5 --batch-size 10000 --insert-ratio 0.5 \
  --duplicate-ratio 0.1 --distribution zipf --mode adaptive \
  --parallel-threshold 1024 --warmup 3 --iterations 10

for kernel in prepare_parallel_merge scatter_parallel_merge; do
  CUDA_VISIBLE_DEVICES="${profile_gpu}" "${ncu_bin}" \
    --set full --kernel-name-base demangled \
    --kernel-name "regex:${kernel}" --launch-count 1 \
    --force-overwrite -o "${result_dir}/update_${kernel}_ncu" \
    "${update_bin}" --vertices 4096 --segment-capacity 2048 \
    --density 0.5 --batch-size 10000 --insert-ratio 0.5 \
    --duplicate-ratio 0.1 --distribution uniform --mode parallel \
    --warmup 0 --iterations 1
done

for mode in atomic warp; do
  CUDA_VISIBLE_DEVICES="${profile_gpu}" "${nsys_bin}" profile \
    --trace=cuda,nvtx,osrt --sample=none --stats=true \
    --force-overwrite=true -o "${result_dir}/delta_${mode}_nsys" \
    "${delta_bin}" --vertices 1024 --block-capacity 32 \
    --batch-size 100000 --input-order grouped --distribution zipf \
    --mode "${mode}" --pool-factor 256 --warmup 3 --iterations 10

  CUDA_VISIBLE_DEVICES="${profile_gpu}" "${ncu_bin}" \
    --set full --replay-mode application --kernel-name-base demangled \
    --kernel-name 'regex:append_delta_kernel' --launch-count 1 \
    --force-overwrite -o "${result_dir}/delta_${mode}_ncu" \
    "${delta_bin}" --vertices 1024 --block-capacity 32 \
    --batch-size 10000 --input-order grouped --distribution zipf \
    --mode "${mode}" --pool-factor 64 --warmup 0 --iterations 1
done

CUDA_VISIBLE_DEVICES="${profile_gpu}" "${nsys_bin}" profile \
  --trace=cuda,nvtx,osrt --sample=none --stats=true \
  --force-overwrite=true -o "${result_dir}/cow_publication_nsys" \
  "${publication_bin}" --segments 256 --capacity 1024 \
  --epochs 32 --repetitions 5
