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
