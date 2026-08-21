# HydraPMA

Hopper-native experiments for dynamic graph storage based on Packed Memory
Arrays (PMA).  The first milestone isolates PMA segment redistribution and
measures where scalar cooperative copies, Ampere-style `cp.async`, and Hopper
bulk asynchronous copies cross over.

The repository deliberately separates three layers:

- `src/`: self-contained architecture microbenchmarks and, later, HydraPMA.
- `third_party/LPMA`: pinned public LPMA repository; it also contains GPMA.
- `scripts/` and `configs/`: reproducible build and experiment drivers.

## Current baseline

`third_party/LPMA` is pinned as a Git submodule at commit
`eb22cd4e1515e83ace93868a2f1e9f2b3b6a53ea`.  The PVLDB 2025 paper's aggregate
artifact URL (`pkumod/GPU_DGM`) returned "Repository not found" on 2026-08-17,
so the public LPMA repository is the initial source for both GPMA and LPMA.

## Build on a Hopper system

Requirements: Linux, CMake 3.22+, CUDA 12.x, and a Hopper GPU.

```bash
git submodule update --init --recursive
cmake -S . -B build -DHYDRAPMA_CUDA_ARCH=sm_90a
cmake --build build -j
./build/segment_bench --segment-bytes 4096 --density 0.75
```

Build the unmodified baseline through the compatibility wrapper:

```bash
./scripts/build_lpma_baselines.sh
```

Build the pinned SlabHash comparison on CUDA 12.x / SM90:

```bash
./scripts/build_slabhash_hopper.sh
```

Run the complete first sweep:

```bash
python3 scripts/run_segment_sweep.py \
  --binary ./build/segment_bench \
  --config configs/h100_segment_sweep.json \
  --output results/h100_segment_sweep.csv
```

The benchmark now preserves five ablation modes: `scalar`, `cp_async`,
`tma_bulk`, fixed-footprint `tma_tiled`, and the two-stage producer/consumer
`tma_pipeline`. On H20, the pipeline is intended for 64 KiB and larger
segments; smaller segments retain the lighter paths.

Those five modes intentionally model the compact live-prefix phase after an
update planner has found the live entries. Arbitrary PMA gaps are measured
separately so their scan cost cannot be hidden:

```bash
./build/segment_bench --segment-bytes 65536 --density 0.7 \
  --layout spread --mode gap_all

python3 scripts/run_segment_sweep.py \
  --binary ./build/segment_bench \
  --config configs/h20_gap_sweep.json \
  --output results/h20_gap_sweep.csv
```

`gap_all` preserves four arbitrary-gap ablations. `gap_scan` uses direct
global loads and a stable CTA prefix scan. `gap_tma_pipeline` assigns a
producer warp to double-buffered 16 KiB TMA loads while seven consumer warps
compact and scatter live entries. `gap_tma_chunked` reduces barriers but
intentionally exposes the loss from strided stores. `gap_tma_buffered` adds a
shared compact tile to recover coalesced rank-order scatter. All four paths
clear the output and examine the full input segment inside the timed kernel;
`prefix` and evenly gapped `spread` layouts must produce the same ordered
result.

The benchmark emits measured data only.  It exits instead of silently labeling
a non-Hopper path as TMA when `sm_90a` support is absent.

The first end-to-end one-level update baseline is:

```bash
./build/update_bench --vertices 4096 --segment-capacity 256 \
  --density 0.5 --batch-size 65536 --insert-ratio 0.5 \
  --duplicate-ratio 0.1 --distribution uniform \
  --mode adaptive --parallel-threshold 1024
```

It sorts and resolves conflicting updates on the GPU, builds affected-segment
offsets, merges insertions/deletions into gapped segments, and compares every
result with an independent CPU last-write-wins reference. Its phase-serial
per-segment merge is retained as the small-task baseline. The parallel path
compacts the gapped base with a CTA scan, computes an exclusive prefix of each
update's net degree effect, and scatters base/update entries directly to unique
final ranks. Adaptive mode buckets segments on the GPU using
`segment_capacity + update_count`.

Run the initial semantics matrix with:

```bash
python3 scripts/run_update_sweep.py --binary ./build/update_bench \
  --config configs/h20_update_correctness.json \
  --output results/h20_update_correctness.csv
```

See `docs/h20_initial_results.md` for the first H20 measurements and Nsight
diagnosis, `docs/hopper_dynamic_graph_research.md` for the CPU-to-Hopper research
roadmap, `docs/hydrapma_ideas_technical_update.md` for the consolidated
baseline ideas, kernel details, and current progress, and
`docs/experiment_plan.md` for the original experiment plan. The staged
end-to-end adaptive and non-blocking implementation is specified in
`docs/lockfree_implementation_plan.md`.
