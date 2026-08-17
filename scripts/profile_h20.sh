#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${repo_root}/results"
segment_bin="${repo_root}/build/segment_bench"
slab_bin="${repo_root}/third_party/SlabHash/build-hopper/bin/benchmark"
nsys_bin="${NSYS_BIN:-/usr/local/bin/nsys}"
ncu_bin="${NCU_BIN:-/usr/local/cuda-12.9/bin/ncu}"
profile_gpu="${PROFILE_GPU:-0}"

mkdir -p "${result_dir}"

CUDA_VISIBLE_DEVICES="${profile_gpu}" "${nsys_bin}" profile \
  --trace=cuda,nvtx,osrt --sample=none --stats=true \
  --force-overwrite=true -o "${result_dir}/slabhash_mode3_nsys" \
  "${slab_bin}" -mode 3 -nStart 18 -nEnd 21 -num_batch 4 -init_batch 3 \
  -lf_conc_step 0.1 -lf_conc_num_sample 10 -device 0 -iter 1 -verbose 0 \
  -filename "${result_dir}/slabhash_mode3_nsys.json"

CUDA_VISIBLE_DEVICES="${profile_gpu}" "${ncu_bin}" \
  --target-processes all --set full --kernel-name-base demangled \
  --kernel-name 'regex:batched_operations' --launch-skip 3 --launch-count 1 \
  --force-overwrite -o "${result_dir}/slabhash_mixed_update_ncu" \
  "${slab_bin}" -mode 3 -nStart 18 -nEnd 21 -num_batch 4 -init_batch 3 \
  -lf_conc_num_sample 1 -device 0 -iter 1 -verbose 0 \
  -filename "${result_dir}/slabhash_mixed_update_ncu.json"

for segment_bytes in 8192 16384 32768; do
  for mode in cp_async tma tma_tiled; do
    CUDA_VISIBLE_DEVICES="${profile_gpu}" "${ncu_bin}" \
      --set full --kernel-name-base demangled \
      --kernel-name 'regex:redistribute' --launch-count 1 \
      --force-overwrite \
      -o "${result_dir}/segment_${segment_bytes}_${mode}_ncu" \
      "${segment_bin}" --segment-bytes "${segment_bytes}" --density 0.75 \
      --working-set-mb 16 --warmup 0 --iterations 1 --mode "${mode}"
  done
done

for segment_bytes in 65536 98304; do
  for mode in cp_async tma tma_tiled tma_pipeline; do
    CUDA_VISIBLE_DEVICES="${profile_gpu}" "${ncu_bin}" \
      --set full --kernel-name-base demangled \
      --kernel-name 'regex:redistribute' --launch-count 1 \
      --force-overwrite \
      -o "${result_dir}/segment_${segment_bytes}_${mode}_ncu" \
      "${segment_bin}" --segment-bytes "${segment_bytes}" --density 0.75 \
      --working-set-mb 16 --warmup 0 --iterations 1 --mode "${mode}"
  done
done
