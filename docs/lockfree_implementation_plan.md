# HydraPMA adaptive and lock-free implementation plan

## Objective

Turn the Hopper segment-compaction primitives into an end-to-end dynamic graph
update path, then replace foreground PMA mutation with a non-blocking
base-plus-delta design. Both paths use the same stream, last-write-wins
semantics, CPU reference, and reporting schema.

Research question:

> Can a hotness-adaptive base-plus-delta layout convert fine-grained updates
> into regular Hopper compaction tasks while avoiding segment locks and
> retaining ordered neighbor traversal?

## Problem contract

The first implementation is deliberately limited to:

- one graph resident on one GPU;
- one fixed-capacity adjacency segment per vertex;
- batched insert/delete with last-write-wins duplicates;
- sorted destination IDs after every published batch;
- explicit overflow instead of silent truncation;
- no multi-level PMA propagation or concurrent queries yet.

## Phase A: one-level update baseline

GPU preprocessing:

```text
sort by (src, dst, sequence descending)
    -> unique by (src, dst)
    -> count per segment
    -> exclusive scan to segment offsets
```

One CTA then owns one segment:

```text
scan old PMA gaps
    -> merge base with updates
    -> apply insert/delete
    -> dense scratch
    -> redistribute gaps
```

The first merge is phase-serial inside the CTA. It establishes correctness and
exposes the speedup available to parallel merge and the existing TMA compactor.

Required output:

- preprocessing and update time separately;
- input and unique updates;
- affected and overflowing segments;
- updates/s and approximate bytes per update;
- exact comparison with an independent CPU reference.

Gate: uniform and Zipf streams must pass insert-only, delete-only, and 50/50
mixes with duplicate conflicts at densities 0.5, 0.7, and 0.9 when capacity
permits.

## Phase B: adaptive base plus delta

Candidate layouts:

```text
INLINE          tiny, cold adjacency
PACKED          read-mostly adjacency
PMA             balanced adjacency
BASE_DELTA      update-heavy adjacency
HUB_PARTITIONED contended high-degree adjacency
```

The first adaptive prototype implements only PMA and BASE_DELTA. It uses update
and query EWMA, delta high-water mark, compaction bytes, CAS retries, density,
and degree. Transitions use hysteresis and emit their thresholds with results.

Initial execution policy comes directly from the measured H20 crossover:

```text
small task        direct warp/CTA path
medium task       raw TMA path
large task        buffered producer/consumer path
very large hub    future cluster path
```

Host dispatch is only for bring-up. The target is GPU descriptor bucketing,
then a persistent-CTA experiment.

## Phase C: lock-free foreground updates

An atomic 64-bit handle points to an immutable segment descriptor. Layout or
base changes create a replacement descriptor and publish it with CAS, avoiding
an inconsistent multi-field header.

```cpp
struct SegmentDesc {
  uint64_t base_handle;
  uint64_t active_delta;
  uint64_t sealed_delta;
  uint32_t base_count;
  uint32_t delta_count;
  uint16_t layout_mode;
  uint16_t generation;
};
```

Updates targeting the same partition use one warp-aggregated reservation:

```text
match/group
    -> leader atomic reservation
    -> coalesced record stores
    -> release fence
    -> READY publication
```

Reservation is not the linearization point. A record becomes visible only when
its READY bit is published after its payload.

On overflow, a winning CAS installs a fresh active delta. Losers read the new
handle and retry. The old delta becomes sealed and remains readable. The first
allocator is a preallocated monotonic pool with no reuse, isolating reclamation
and ABA from append correctness.

Copy-on-write compaction is:

```text
freeze old generation
    -> writers switch to a new delta
    -> merge immutable base and sealed delta privately
    -> build replacement base and descriptor
    -> CAS active descriptor handle
    -> retire old objects
```

Claims are staged:

1. no mutex or segment spinlock;
2. compaction does not block foreground update/query;
3. strict lock-free progress only after pool exhaustion, retry starvation,
   ABA, and reclamation are handled.

## Phase D: publication and queries

Queries acquire a descriptor handle and validate its generation after
traversal. The first version retries on change. The next uses segment read
epochs and reclaims old allocations only after older readers leave. Stress
tests must show that queries observe an old or new snapshot, never a partial
segment.

## Experimental matrix

Compare:

1. phase-serial one-level PMA;
2. adaptive direct/TMA PMA;
3. per-edge atomic delta append;
4. warp-aggregated lock-free append;
5. lock-free append plus asynchronous compaction.

Sweep batch size, uniform/Zipf locality, insert/delete mix, density,
update/query ratio, and delta partition count. Report committed updates/s,
neighbor edges/s, batch latency distribution, atomics and CAS retries per
update, query retries, bytes per update, delta query amplification, compaction
duty cycle, and the relevant NCU stall/traffic metrics.

## Planned commits

1. `Add one-level dynamic graph update benchmark`
2. `Parallelize update merge and add adaptive dispatch`
3. `Add warp-aggregated lock-free delta append`
4. `Add versioned delta rollover and COW publication`
5. `Integrate Hopper compaction and concurrent query validation`
6. `Profile adaptive lock-free updates on H20`

## Implementation status: 2026-08-21

Completed and compiled for CUDA 12.9 / `sm_90a`:

- `09bb60c`: one-level GPU preprocessing, insert/delete merge, PMA
  redistribution, and independent CPU reference;
- `2059ed9`: parallel effect-prefix merge and GPU serial/parallel adaptive
  bucketing;
- `4c39899`: per-edge versus warp-aggregated delta reservation, READY
  publication, generation-tagged handles, CAS rollover, and exactly-once chain
  validation;
- `67e433e`: immutable replacement descriptors, versioned COW publication,
  and concurrent snapshot validation.

H20 runtime validation on 2026-08-21 completed 126 one-level update rows,
260 paired delta rows, and 26 COW publication rows with no correctness
failures after sizing the monotonic delta pool for measured CAS-loser waste.
Warp aggregation wins all Zipf points, with median speedups of 1.566x for
grouped input and 1.230x for random input, but uniform/random has a 0.986x
median and therefore requires adaptive dispatch.

The original persistent-reader publication test did not have a portable
cross-stream forward-progress guarantee on the H20 driver. It now launches a
finite snapshot kernel per publication epoch and preserves overlap through a
high-priority query stream. The full sweep reports zero partial snapshots and
zero publication CAS failures. Epoch reclamation and integrated asynchronous
compaction are still pending, so this is not yet a complete lock-free graph
store.

The first adaptive delta dispatcher is also complete. It combines estimated
within-warp source grouping, hottest-source fraction, and a large-batch
threshold. The final H20 matrix has zero failures and selects the faster path
for all 17 configurations whose five-seed medians differ by at least 2%.
Feature extraction is still host-side in the synthetic harness; moving these
statistics into the GPU planner is required before an end-to-end claim.
