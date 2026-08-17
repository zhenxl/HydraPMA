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

The benchmark emits measured data only.  It exits instead of silently labeling
a non-Hopper path as TMA when `sm_90a` support is absent.

See `docs/h20_initial_results.md` for the first H20 measurements and Nsight
diagnosis, and `docs/experiment_plan.md` for hypotheses and follow-up work.
