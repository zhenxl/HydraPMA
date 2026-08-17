# Baseline inventory

## Included now

| Baseline | Source | Pin | Role |
|---|---|---|---|
| GPMA | `third_party/LPMA/gpma.cuh` | `eb22cd4e...` | Classical contiguous PMA |
| LPMA/RPMA | `third_party/LPMA/rpma.cuh` | `eb22cd4e...` | Leveled/redundancy-reduced PMA |
| SlabHash | `third_party/SlabHash` | `eec7135a...` | GPU hash-table contrast |

The public LPMA repository is small and contains the authors' GPMA comparison
implementation, update drivers, and rebalance-length drivers.  We keep it as an
unmodified submodule and adapt its build from outside the directory.

SlabHash is also kept as an unmodified submodule. The recorded
`patches/slabhash-sm90-cuda12.patch` only adds an SM90 CMake target and avoids
building its historical GoogleTest dependency when tests are disabled.
`scripts/build_slabhash_hopper.sh` applies that patch reproducibly.

## Hopper compatibility result

The public GPMA/LPMA source uses CUDA Dynamic Parallelism CDP1 and calls
`cudaDeviceSynchronize()` from device code. CUDA 12.9 can compile that call
only for pre-SM90 PTX, and loading the resulting PTX on H20 fails with
`cudaErrorUnsupportedDevSideSync`. Removing the synchronization changes the
algorithm, so the unmodified implementation is recorded as incompatible rather
than reported as an H20 performance result.

SlabHash compiles and runs natively as `sm_90` with CUDA 12.9. This gives the
initial end-to-end update baseline while a host-orchestrated native GPMA port is
implemented.
## Reproducibility issue

The artifact link printed by *Towards Sufficient GPU-accelerated Dynamic Graph
Management* (PVLDB 18(3), 2025), `https://github.com/pkumod/GPU_DGM`, returned
"Repository not found" when cloned on 2026-08-17.  Do not claim results from the
six-system artifact unless the exact code is recovered from the authors or an
archive.

## Planned additions after the microbenchmark gate

1. Rebuild-to-CSR using CUB sort/scan as the compact static-layout baseline.
2. Optionally Hornet/faimGraph if their historical CUDA dependencies can be
   containerized without source-level semantic changes.
3. A host-orchestrated GPMA port that preserves level propagation semantics and
   is explicitly labeled as a port rather than the unmodified baseline.

Every added baseline must record repository URL, exact commit, CUDA version,
patches, and whether correctness tests pass.
