# H20 initial results

Date: 2026-08-17. Device: NVIDIA H20, compute capability 9.0, 78 SMs.
Toolkit: CUDA 12.9. Driver: 575.57.08. Measurements used one otherwise idle
GPU. Raw CSV, JSON, `.nsys-rep`, SQLite, and `.ncu-rep` files are retained
under the ignored local `results/` directory.

## Baseline status

- The self-contained segment redistribution benchmark passes validation for
  scalar, `cp.async`, and Hopper bulk-async paths.
- SlabHash at commit `eec7135a...` builds as native `sm_90` after the
  recorded build-system-only compatibility patch.
- The unmodified public GPMA/LPMA baseline cannot run on SM90. Its CDP1
  device-side `cudaDeviceSynchronize()` fails at module load with
  `cudaErrorUnsupportedDevSideSync`. This is a compatibility result, not a
  performance result.

## Segment sweep

The complete sweep contains 450 raw mode rows: 10 segment sizes, 3 densities,
5 process repetitions, and 3 copy modes. All validation fields are true.

`cp.async` wins most points. The steady-state median shows a narrow TMA win
only around 16 KiB (about 2.3-2.4% versus the best non-TMA path at the three
tested densities). TMA loses again at 32 KiB and above. The current bulk path
uses a maximum 16 KiB transaction, so a 32 KiB segment needs two load and two
store transactions.

Do not use application-reported latency printed while NCU is replaying a
kernel. The normal sweep is the performance source of truth; NCU is used for
counter diagnosis.

## SlabHash baseline

Mode 0, 4,194,304 keys and queries, expected chain 0.6, five iterations:

- measured load factor: 0.55
- build rate: 27.81 M elements/s
- singleton search rate: 10,588.08 M queries/s
- bulk search rate: 12,355.96 M queries/s

Mode 3 uses 256 Ki-operation batches, three initial insertion batches and one
mixed batch. Across expected-chain samples 0.1 through 1.0, the reported mixed
rate ranges from 5,027.82 to 7,668.02 M operations/s.

## Nsight Systems: end-to-end SlabHash

The mode-3 trace shows orchestration and initialization dominate:

| Category | Count | GPU/API time |
|---|---:|---:|
| update kernels | 40 | 1.866 ms |
| bucket-count kernels | 20 | 0.798 ms |
| CUDA memsets | 730 | 4.303 ms |
| D2H copies | 80 | 5.499 ms |
| H2D copies | 40 | 1.031 ms |
| `cudaMalloc` API | 110 | 46.834 ms |
| `cudaFree` API | 110 | 20.280 ms |

The driver also spends roughly 20.4 s in host `poll` calls during the profiled
run. Much of this is benchmark-side random batch generation and per-sample
allocation/initialization, so it must be separated from steady-state data
structure throughput in future comparisons.

## Nsight Compute: SlabHash mixed update

One 262,144-operation mixed-update launch:

| Metric | Value |
|---|---:|
| kernel duration | 42.94 us |
| compute throughput | 42.96% |
| DRAM throughput | 18.61% (748.23 GB/s) |
| L1 hit rate | 0.11% |
| L2 hit rate | 15.52% |
| achieved occupancy | 78.76% |
| branch efficiency | 95.15% |
| scheduler cycles with no eligible warp | 53.05% |
| long-scoreboard stall | 17.65 cycles/issue |

The dominant limitation is dependent, irregular global-memory latency. It is
not peak HBM bandwidth, low theoretical occupancy, or branch divergence. This
supports using SlabHash as the random-access contrast to a contiguous PMA, but
also motivates batching, bucket grouping, and software prefetch for the hash
path.

## Nsight Compute: PMA segment redistribution

Density is 0.75 and the working set is 16 MiB. NCU durations below are
single-profile-launch counters, not the steady-state sweep.

| Segment | Path | NCU duration | DRAM peak | Active warps | Long scoreboard | Async instruction count |
|---:|---|---:|---:|---:|---:|---:|
| 8 KiB | `cp.async` | 11.14 us | 38.52% | 87.56% | 7.63 | 32,768 LDGSTS |
| 8 KiB | TMA bulk | 13.15 us | 33.01% | 72.64% | 15.68 | 2,048 load + 2,048 store |
| 16 KiB | `cp.async` | 12.67 us | 33.98% | 64.06% | 5.37 | 32,768 LDGSTS |
| 16 KiB | TMA bulk | 13.79 us | 31.42% | 49.61% | 14.91 | 1,024 load + 1,024 store |
| 32 KiB | `cp.async` | 13.89 us | 30.59% | 33.51% | 4.18 | 32,768 LDGSTS |
| 32 KiB | TMA bulk | 17.79 us | 24.20% | 24.14% | 11.90 | 1,024 load + 1,024 store |

TMA-pipe utilization is only 0.18-0.55%. Double-buffer shared memory grows
from 16 to 32 to 64 KiB per block, and the shared-memory block limit drops from
9 to 6 to 3. The current kernel is phase-serial: one elected thread issues the
whole bulk load, all threads wait and redistribute, then the elected thread
issues the whole bulk store. It therefore exposes TMA latency instead of
overlapping it.

## Research decision

The next prototype should use a persistent CTA with a dedicated producer warp,
consumer warps, two or three fixed-size tiles, and per-stage `mbarrier`
handoff. Segment size should not determine shared-memory footprint. A
level-aware dispatcher can retain `cp.async` for small/medium rebalances and
route only sufficiently large or coalesced regions through the pipelined TMA
path. End-to-end work should also reuse allocations and batch bucket/segment
metadata on device to remove the initialization wall exposed by Nsight
Systems.

## Dynamic update and lock-free follow-up

Date: 2026-08-21. Device: one NVIDIA H20 on the new Shanghai node
`118.196.122.59`, CUDA 12.9, native `sm_90a`. GPU 1 was isolated from an
unrelated 84.5 GiB allocation on GPU 0. The source baseline was
`dc82426`, followed by the publication forward-progress correction described
below.

Raw result files retained under `results/`:

- `h20_update_correctness_20260821.csv`;
- `h20_lockfree_delta_sweep_20260821.csv`;
- `h20_publication_sweep_20260821.csv`;
- four exported Nsight Compute CSV reports for update prepare/scatter and
  delta atomic/warp.

### Bring-up corrections

The original publication microbenchmark used a persistent reader in one
stream and a publisher that waited for a reader-start flag in another. On this
driver, the publisher could occupy scheduling resources before the persistent
reader made progress. The benchmark was changed to submit one finite snapshot
kernel per epoch. The publisher and high-priority query stream still overlap,
but both kernels terminate independently. A 256-segment, 256-entry,
32-epoch run produced 1,787 handle-change retries, zero mismatches, and zero
CAS failures, demonstrating actual overlap without relying on cross-kernel
forward progress.

The monotonic delta allocator intentionally does not reclaim blocks lost by a
rollover CAS. The original pool factors therefore exhausted under contention.
Correctness sweeps now allocate factor 64 for uniform input and factor 256 for
Zipf input. Pool initialization is outside the timed append interval.

### Correctness matrices

| Matrix | Measurement rows | Incorrect | Pool/segment overflow |
|---|---:|---:|---:|
| one-level adaptive update | 126 | 0 | 0 |
| atomic/warp delta append | 260 | 0 | 0 |
| COW publication | 26 | 0 | 0 |

### One-level update

Uniform cases use 512-entry segments and all 108 rows route to the serial
segment kernel. Zipf cases use 2,048-entry segments and all 18 rows route to
the parallel path. This exposes a dispatcher problem: the current
`segment_capacity + update_count` threshold is dominated by capacity rather
than measured work or contention.

| Distribution / input batch | Median update MUPS | Median preprocess | Median update |
|---|---:|---:|---:|
| uniform / 1,024 | 4.146 | 0.2102 ms | 0.2161 ms |
| uniform / 10,000 | 10.625 | 0.2536 ms | 0.8209 ms |
| Zipf / 1,024 | 37.816 | 0.2157 ms | 0.0203 ms |
| Zipf / 5,000 | 75.959 | 0.2441 ms | 0.0492 ms |

Preprocessing is already the latency floor for the fast Zipf/parallel cases.
The throughput values are update-phase MUPS; end-to-end latency must add
`preprocess_ms`.

### Warp-aggregated delta append

Across 130 paired points, warp aggregation wins 92 and has median speedup
1.030x. The aggregate number hides a strong locality crossover:

| Distribution / order | Pairs | Warp wins | Median speedup | Warp/atomic reservation ratio |
|---|---:|---:|---:|---:|
| uniform / grouped | 45 | 31 | 1.019x | 0.130 |
| uniform / random | 45 | 21 | 0.986x | 0.985 |
| Zipf / grouped | 20 | 20 | 1.566x | 0.045 |
| Zipf / random | 20 | 20 | 1.230x | 0.369 |

At 100K updates, Zipf/grouped reaches 1.732x for 32-entry blocks and 1.750x
for 64-entry blocks. Uniform/random remains near break-even because
`match_any` finds almost no same-source lanes.

Nsight Compute on the 10K Zipf/grouped, 32-entry point:

| Metric | Per-edge atomic | Warp aggregated |
|---|---:|---:|
| duration | 165.15 us | 115.23 us |
| scheduler cycles with no eligible warp | 98.00% | 96.70% |
| warp cycles per issued instruction | 83.59 | 49.43 |
| achieved occupancy | 9.00% | 8.42% |
| average active threads per warp | 29.22 | 11.82 |
| L1 hit rate | 8.55% | 21.49% |
| L2 hit rate | 99.43% | 95.29% |

The improvement comes from reducing serialized reservation/rollover work, not
from bandwidth or occupancy. The remaining bottlenecks are a small 40-block
grid on a 78-SM GPU, very low eligible-warp rate, and load imbalance around hot
vertices.

### COW publication

All accepted snapshots match a complete immutable generation. Across the 26
points there are zero partial-generation mismatches and zero publication CAS
failures.

| Replacement capacity | Median publication rate | Median reader retry rate |
|---:|---:|---:|
| 64 | 41.456 M/s | 2.277% |
| 256 | 42.725 M/s | 2.202% |
| 1,024 | 26.427 M/s | 10.709% |
| 4,096 | 25.382 M/s | 19.417% |

Large replacement construction increases the overlap window and therefore
reader retries. This is the first measured query-side cost of COW publication.

### Nsight diagnosis for parallel merge

Nsight Systems shows `prepare_parallel_merge_kernel` and
`scatter_parallel_merge_kernel` averaging about 42.7 and 44.5 us,
respectively, and together consuming about 88% of traced GPU kernel time.
Full NCU sampling gives:

| Metric | Prepare | Scatter |
|---|---:|---:|
| NCU duration | 81.50 us | 79.01 us |
| memory throughput | 54.46% | 77.75% |
| DRAM throughput | 50.95% | 51.81% |
| L1 hit rate | 32.95% | 9.88% |
| L2 hit rate | 39.15% | 79.76% |
| achieved occupancy | 94.86% | 83.97% |
| scheduler cycles with no eligible warp | 67.43% | 65.30% |

Occupancy is already high and scatter is cache/memory-pipeline limited. The
next merge optimization should fuse preparation and scatter, or keep compacted
base data in CTA-local storage, to remove the intermediate global-memory
round trip. Increasing occupancy alone is not supported by these counters.

### Next measured experiments

1. Add a batch-level adaptive delta dispatcher. Use sampled within-warp
   same-source density and batch size to bypass `match_any` for
   uniform/random or small batches.
2. Prototype a fused parallel merge for segments that fit a bounded CTA tile;
   retain the current two-kernel path as the large-segment fallback.
3. Reclaim CAS-loser delta blocks with a generation-tagged free list, then
   reduce the 64/256 overprovisioning factors and measure allocator traffic.
4. Extend publication measurements with query amplification and epoch-based
   reclamation before claiming full lock-free storage.
