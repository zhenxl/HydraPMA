# Initial experiment plan

## Research gate

The first experiment asks a narrow question:

> At what PMA segment sizes and densities does Hopper bulk-asynchronous segment
> movement amortize its synchronization and shared-memory costs relative to
> scalar cooperative copies and `cp.async`?

The benchmark performs the PMA redistribution primitive: it loads a segment
whose live entries are compacted at the front, places those entries at evenly
spaced destinations, and writes the gapped segment back.  This is intentionally
not yet a full insertion/deletion benchmark.

## Independent variables

- Segment bytes: 256 B through 96 KiB.
- Density: 0.5, 0.7, and 0.9.
- Movement backend: scalar, `cp.async`, Hopper bulk async (TMA engine).
- Update locality in the next milestone: uniform segment selection versus
  Zipf/hotspot selection.

Each point runs five process-level repetitions.  Within a process, five warmups
and twenty measured iterations operate on a 64 MiB working set.  The two shared
buffers make 96 KiB segments consume 192 KiB of shared memory and fit H100's
opt-in per-block limit.

## Dependent variables

- Kernel latency and effective read+write GB/s.
- Correctness of the complete first output segment after every mode.
- Follow-up profiler metrics: DRAM throughput, L2 hit rate, active warps,
  barrier stalls, TMA requests, and achieved occupancy.

## Controls and cautions

- All modes execute identical gap-placement arithmetic and write the entire
  output segment, including gaps.
- The TMA backend uses 16 KiB chunks because bulk-copy transaction sizes are
  bounded; larger PMA segments issue multiple transactions against one
  transaction barrier.
- Small-segment results include one-block-per-segment scheduling overhead.  This
  is desired for the adaptive-path decision but must not be presented as a pure
  copy-bandwidth result.
- Verify generated SASS contains `CPASYNC` for `cp_async` and bulk/TMA
  instructions for `tma_bulk` before interpreting performance.
- The current benchmark is single-stage, not yet double buffered.  It establishes
  a lower-risk break-even curve before a persistent warp-specialized pipeline.

## Pass/fail decision

Proceed to a warp-specialized GPMA prototype if at least one practically common
segment range shows a stable bulk-async advantage, or if profiling shows that
movement is sufficiently hidden to reduce end-to-end redistribution time.

If TMA never wins, retain the measured negative result and pivot to the still
useful level-aware dispatcher: warp path for small segments, `cp.async` CTA path
for medium segments, and multi-CTA planning for large segments.

## Milestone 2: baseline comparison

1. Generate deterministic RMAT and power-law edge streams with uniform and
   hotspot batches.
2. Run GPMA and LPMA at batch sizes 1K, 10K, 100K, and 1M.
3. Record per-batch latency distributions, not only averages.
4. Instrument rebalance bytes and selected levels in the baseline.
5. Replace only the GPMA redistribution primitive with the winning backend and
   measure end-to-end speedup.

## Milestone 3: research prototype

- Pre-plan non-overlapping rebalance regions for each batch.
- Dispatch small/medium/large regions to warp/CTA/cluster paths.
- Add double-buffered producer/consumer stages with `mbarrier`.
- Add versioned region publication and measure mixed query/update workloads.

