#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct alignas(16) Entry {
  std::uint64_t key;
  std::uint64_t value;
};

static_assert(sizeof(Entry) == 16, "bulk copies require 16-byte entries");

constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
constexpr int kThreads = 256;
constexpr int kMaxBulkBytes = 16 * 1024;
constexpr std::size_t kTileEntries = kMaxBulkBytes / sizeof(Entry);
constexpr int kProducerThreads = 32;
constexpr int kConsumerThreads = kThreads - kProducerThreads;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(status_));                     \
    }                                                                            \
  } while (0)

enum class Mode {
  kScalar,
  kCpAsync,
  kTma,
  kTmaTiled,
  kTmaPipeline,
  kGapScan,
  kGapTmaPipeline,
  kGapTmaChunked,
  kGapTmaBuffered
};

enum class Layout { kPrefix, kSpread };

struct Options {
  std::size_t segment_bytes = 4096;
  double density = 0.75;
  std::size_t working_set_mb = 64;
  int warmup = 5;
  int iterations = 20;
  std::string mode = "all";
  std::string layout = "prefix";
};

__device__ __forceinline__ std::uint32_t smem_address(const void* ptr) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t dst = smem_address(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :
               : "r"(dst), "l"(gmem)
               : "memory");
#endif
}

__device__ __forceinline__ void cp_async_commit_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbarrier_init(std::uint64_t* barrier) {
  const std::uint32_t addr = smem_address(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" : : "r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    std::uint64_t* barrier, std::uint32_t bytes) {
  const std::uint32_t addr = smem_address(barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(addr), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(std::uint64_t* barrier,
                                               int phase) {
  const std::uint32_t addr = smem_address(barrier);
  asm volatile(
      "{\n"
      ".reg .pred ready;\n"
      "hydrapma_wait:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1;\n"
      "@!ready bra hydrapma_wait;\n"
      "}\n"
      :
      : "r"(addr), "r"(phase)
      : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* smem, const void* gmem,
                                         int bytes,
                                         std::uint64_t* barrier) {
  const std::uint32_t dst = smem_address(smem);
  const std::uint32_t bar = smem_address(barrier);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n"
      :
      : "r"(dst), "l"(gmem), "r"(bytes), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void bulk_s2g(void* gmem, const void* smem,
                                         int bytes) {
  const std::uint32_t src = smem_address(smem);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
               :
               : "l"(gmem), "r"(src), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void consumer_barrier() {
  asm volatile("bar.sync 1, %0;\n"
               :
               : "r"(kConsumerThreads)
               : "memory");
}
#endif

__host__ __device__ __forceinline__ std::size_t redistributed_position(
    std::size_t live_index, std::size_t capacity, std::size_t live_count) {
  return (live_index * capacity) / live_count;
}

__host__ __device__ __forceinline__ std::size_t ceil_div(
    std::size_t numerator, std::size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

__global__ void initialize_input(Entry* input, Entry* output,
                                 std::size_t total_entries,
                                 std::size_t capacity,
                                 std::size_t live_count, Layout layout) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < total_entries; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t local = i % capacity;
    std::size_t live_index = live_count;
    if (layout == Layout::kPrefix) {
      if (local < live_count) {
        live_index = local;
      }
    } else {
      const std::size_t candidate = ceil_div(local * live_count, capacity);
      if (candidate < live_count &&
          redistributed_position(candidate, capacity, live_count) == local) {
        live_index = candidate;
      }
    }
    if (live_index < live_count) {
      const std::uint64_t key = live_index + 1;
      input[i] = Entry{key, key ^ 0x5a5a5a5aULL};
    } else {
      input[i] = Entry{kEmpty, 0};
    }
    output[i] = Entry{kEmpty, 0};
  }
}

template <Mode mode>
__global__ void redistribute_kernel(const Entry* input, Entry* output,
                                    std::size_t capacity,
                                    std::size_t live_count) {
  extern __shared__ __align__(16) unsigned char storage[];
  Entry* stage_in = reinterpret_cast<Entry*>(storage);
  Entry* stage_out = stage_in + capacity;
  const Entry* segment_in = input + std::size_t(blockIdx.x) * capacity;
  Entry* segment_out = output + std::size_t(blockIdx.x) * capacity;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ __align__(8) std::uint64_t barrier;
#endif

  for (std::size_t i = threadIdx.x; i < capacity; i += blockDim.x) {
    stage_out[i] = Entry{kEmpty, 0};
  }

  if constexpr (mode == Mode::kScalar) {
    for (std::size_t i = threadIdx.x; i < capacity; i += blockDim.x) {
      stage_in[i] = segment_in[i];
    }
  } else if constexpr (mode == Mode::kCpAsync) {
    for (std::size_t i = threadIdx.x; i < capacity; i += blockDim.x) {
      cp_async_16(stage_in + i, segment_in + i);
    }
    cp_async_commit_wait();
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0) {
      mbarrier_init(&barrier);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int total_bytes = static_cast<int>(capacity * sizeof(Entry));
      mbarrier_arrive_expect_tx(&barrier, total_bytes);
      for (int offset = 0; offset < total_bytes; offset += kMaxBulkBytes) {
        const int chunk = min(kMaxBulkBytes, total_bytes - offset);
        bulk_g2s(reinterpret_cast<unsigned char*>(stage_in) + offset,
                 reinterpret_cast<const unsigned char*>(segment_in) + offset,
                 chunk, &barrier);
      }
    }
    mbarrier_wait(&barrier, 0);
#endif
  }

  __syncthreads();
  for (std::size_t i = threadIdx.x; i < live_count; i += blockDim.x) {
    const std::size_t dst = redistributed_position(i, capacity, live_count);
    stage_out[dst] = stage_in[i];
  }
  if constexpr (mode == Mode::kTma) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Every producer thread must publish its generic shared-memory writes to
    // the async proxy before the elected thread starts the bulk store.
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      const int total_bytes = static_cast<int>(capacity * sizeof(Entry));
      for (int offset = 0; offset < total_bytes; offset += kMaxBulkBytes) {
        const int chunk = min(kMaxBulkBytes, total_bytes - offset);
        bulk_s2g(reinterpret_cast<unsigned char*>(segment_out) + offset,
                 reinterpret_cast<const unsigned char*>(stage_out) + offset,
                 chunk);
      }
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
    }
#endif
  } else {
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < capacity; i += blockDim.x) {
      segment_out[i] = stage_out[i];
    }
  }
}

// Fixed-footprint TMA path. Each CTA walks one PMA segment as a sequence of
// output tiles. The compact live prefix needed by an output tile is contiguous,
// so one bulk load feeds a cooperative gap-placement pass and one bulk store.
// This is intentionally a single-stage ablation: it isolates the occupancy
// effect before producer/consumer overlap is added.
__global__ void redistribute_tma_tiled_kernel(const Entry* input, Entry* output,
                                              std::size_t capacity,
                                              std::size_t live_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(16) unsigned char storage[];
  Entry* stage_in = reinterpret_cast<Entry*>(storage);
  Entry* stage_out = stage_in + kTileEntries;
  const Entry* segment_in = input + std::size_t(blockIdx.x) * capacity;
  Entry* segment_out = output + std::size_t(blockIdx.x) * capacity;
  __shared__ __align__(8) std::uint64_t barrier;

  for (std::size_t output_begin = 0; output_begin < capacity;
       output_begin += kTileEntries) {
    const std::size_t output_end =
        min(capacity, output_begin + kTileEntries);
    const std::size_t output_entries = output_end - output_begin;
    const std::size_t input_begin =
        ceil_div(output_begin * live_count, capacity);
    const std::size_t input_end =
        min(live_count, ceil_div(output_end * live_count, capacity));
    const std::size_t input_entries = input_end - input_begin;

    for (std::size_t i = threadIdx.x; i < output_entries;
         i += blockDim.x) {
      stage_out[i] = Entry{kEmpty, 0};
    }
    if (threadIdx.x == 0) {
      mbarrier_init(&barrier);
    }
    __syncthreads();

    if (input_entries != 0) {
      if (threadIdx.x == 0) {
        const int input_bytes =
            static_cast<int>(input_entries * sizeof(Entry));
        mbarrier_arrive_expect_tx(&barrier, input_bytes);
        bulk_g2s(stage_in, segment_in + input_begin, input_bytes, &barrier);
      }
      mbarrier_wait(&barrier, 0);
    }
    __syncthreads();

    for (std::size_t local = threadIdx.x; local < input_entries;
         local += blockDim.x) {
      const std::size_t live_index = input_begin + local;
      const std::size_t destination =
          redistributed_position(live_index, capacity, live_count);
      stage_out[destination - output_begin] = stage_in[local];
    }

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      const int output_bytes =
          static_cast<int>(output_entries * sizeof(Entry));
      bulk_s2g(segment_out + output_begin, stage_out, output_bytes);
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
    }
    __syncthreads();
  }
#endif
}

// Two-stage producer/consumer pipeline. Thread 0 owns TMA issue and publication;
// the other seven warps cooperatively clear and redistribute a ready tile.
// While consumers work on tile n, the producer can store tile n-1 and keep the
// load for tile n+1 in flight.
__global__ void redistribute_tma_pipeline_kernel(
    const Entry* input, Entry* output, std::size_t capacity,
    std::size_t live_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(16) unsigned char storage[];
  Entry* entries = reinterpret_cast<Entry*>(storage);
  constexpr std::size_t kStageEntries = 2 * kTileEntries;
  const Entry* segment_in = input + std::size_t(blockIdx.x) * capacity;
  Entry* segment_out = output + std::size_t(blockIdx.x) * capacity;
  __shared__ __align__(8) std::uint64_t load_barrier[2];
  __shared__ int compute_ready[2];
  __shared__ std::size_t input_begin[2];
  __shared__ std::size_t input_entries[2];
  __shared__ std::size_t output_begin[2];
  __shared__ std::size_t output_entries[2];

  if (threadIdx.x == 0) {
    mbarrier_init(&load_barrier[0]);
    mbarrier_init(&load_barrier[1]);
    compute_ready[0] = 0;
    compute_ready[1] = 0;
  }
  __syncthreads();

  const std::size_t tile_count = ceil_div(capacity, kTileEntries);
  if (threadIdx.x == 0) {
    const std::size_t preload = tile_count < 2 ? tile_count : 2;
    for (std::size_t tile = 0; tile < preload; ++tile) {
      const int stage = static_cast<int>(tile & 1);
      const std::size_t begin = tile * kTileEntries;
      const std::size_t end = min(capacity, begin + kTileEntries);
      output_begin[stage] = begin;
      output_entries[stage] = end - begin;
      input_begin[stage] = ceil_div(begin * live_count, capacity);
      const std::size_t input_end =
          min(live_count, ceil_div(end * live_count, capacity));
      input_entries[stage] = input_end - input_begin[stage];
      __threadfence_block();
      const int bytes =
          static_cast<int>(input_entries[stage] * sizeof(Entry));
      mbarrier_arrive_expect_tx(&load_barrier[stage], bytes);
      if (bytes != 0) {
        bulk_g2s(entries + stage * kStageEntries,
                 segment_in + input_begin[stage], bytes,
                 &load_barrier[stage]);
      }
    }

    for (std::size_t tile = 0; tile < tile_count; ++tile) {
      const int stage = static_cast<int>(tile & 1);
      while (atomicAdd(&compute_ready[stage], 0) <
             static_cast<int>(tile + 1)) {
        __nanosleep(64);
      }

      Entry* stage_out =
          entries + stage * kStageEntries + kTileEntries;
      const int bytes =
          static_cast<int>(output_entries[stage] * sizeof(Entry));
      bulk_s2g(segment_out + output_begin[stage], stage_out, bytes);
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");

      const std::size_t next_tile = tile + 2;
      if (next_tile < tile_count) {
        const std::size_t begin = next_tile * kTileEntries;
        const std::size_t end = min(capacity, begin + kTileEntries);
        output_begin[stage] = begin;
        output_entries[stage] = end - begin;
        input_begin[stage] = ceil_div(begin * live_count, capacity);
        const std::size_t input_end =
            min(live_count, ceil_div(end * live_count, capacity));
        input_entries[stage] = input_end - input_begin[stage];
        __threadfence_block();
        const int input_bytes =
            static_cast<int>(input_entries[stage] * sizeof(Entry));
        mbarrier_arrive_expect_tx(&load_barrier[stage], input_bytes);
        if (input_bytes != 0) {
          bulk_g2s(entries + stage * kStageEntries,
                   segment_in + input_begin[stage], input_bytes,
                   &load_barrier[stage]);
        }
      }
    }
  } else if (threadIdx.x >= kProducerThreads) {
    const int consumer_id = threadIdx.x - kProducerThreads;
    for (std::size_t tile = 0; tile < tile_count; ++tile) {
      const int stage = static_cast<int>(tile & 1);
      const int phase = static_cast<int>((tile / 2) & 1);
      mbarrier_wait(&load_barrier[stage], phase);

      Entry* stage_in = entries + stage * kStageEntries;
      Entry* stage_out = stage_in + kTileEntries;
      const std::size_t out_count = output_entries[stage];
      for (std::size_t i = consumer_id; i < out_count;
           i += kConsumerThreads) {
        stage_out[i] = Entry{kEmpty, 0};
      }
      consumer_barrier();

      const std::size_t in_begin = input_begin[stage];
      const std::size_t in_count = input_entries[stage];
      const std::size_t out_begin = output_begin[stage];
      for (std::size_t local = consumer_id; local < in_count;
           local += kConsumerThreads) {
        const std::size_t live_index = in_begin + local;
        const std::size_t destination =
            redistributed_position(live_index, capacity, live_count);
        stage_out[destination - out_begin] = stage_in[local];
      }
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_barrier();
      if (threadIdx.x == kProducerThreads) {
        atomicExch(&compute_ready[stage], static_cast<int>(tile + 1));
      }
    }
  }
#endif
}

// Arbitrary-gap baseline. A CTA clears one output segment, scans the input in
// 256-entry batches, and uses warp ballots plus an eight-warp prefix to assign
// stable live ranks. Unlike the compact-prefix kernels above, every input slot
// is examined and the scan cost is included in the timed region.
__global__ void redistribute_gap_scan_kernel(const Entry* input, Entry* output,
                                             std::size_t capacity,
                                             std::size_t live_count) {
  const Entry* segment_in = input + std::size_t(blockIdx.x) * capacity;
  Entry* segment_out = output + std::size_t(blockIdx.x) * capacity;
  __shared__ int warp_counts[8];
  __shared__ int warp_offsets[8];
  __shared__ int batch_total;

  for (std::size_t i = threadIdx.x; i < capacity; i += blockDim.x) {
    segment_out[i] = Entry{kEmpty, 0};
  }
  __syncthreads();

  std::size_t live_base = 0;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (std::size_t begin = 0; begin < capacity; begin += kThreads) {
    const std::size_t index = begin + threadIdx.x;
    const Entry entry =
        index < capacity ? segment_in[index] : Entry{kEmpty, 0};
    const bool valid = index < capacity && entry.key != kEmpty;
    const unsigned mask = __ballot_sync(0xffffffffu, valid);
    if (lane == 0) {
      warp_counts[warp] = __popc(mask);
    }
    __syncthreads();

    if (warp == 0) {
      if (lane < 8) {
        int offset = 0;
        for (int i = 0; i < lane; ++i) {
          offset += warp_counts[i];
        }
        warp_offsets[lane] = offset;
      }
      if (lane == 0) {
        int total = 0;
        for (int i = 0; i < 8; ++i) {
          total += warp_counts[i];
        }
        batch_total = total;
      }
    }
    __syncthreads();

    if (valid) {
      const unsigned lower_mask =
          lane == 0 ? 0u : ((1u << lane) - 1u);
      const std::size_t rank =
          live_base + warp_offsets[warp] + __popc(mask & lower_mask);
      const std::size_t destination =
          redistributed_position(rank, capacity, live_count);
      segment_out[destination] = entry;
    }
    __syncthreads();
    live_base += batch_total;
  }
}

// Hopper arbitrary-gap path. The producer warp owns two alternating 16 KiB TMA
// input stages. Seven consumer warps perform a stable ballot compaction and
// scatter live entries directly to their PMA destinations. Output clearing and
// full-capacity gap scanning are intentionally part of the measurement.
template <int strategy>
__global__ void redistribute_gap_tma_pipeline_kernel(
    const Entry* input, Entry* output, std::size_t capacity,
    std::size_t live_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(16) unsigned char storage[];
  Entry* entries = reinterpret_cast<Entry*>(storage);
  Entry* compact = entries + 2 * kTileEntries;
  const Entry* segment_in = input + std::size_t(blockIdx.x) * capacity;
  Entry* segment_out = output + std::size_t(blockIdx.x) * capacity;
  __shared__ __align__(8) std::uint64_t load_barrier[2];
  __shared__ int scan_ready[2];
  __shared__ std::size_t input_entries[2];
  __shared__ int warp_counts[7];
  __shared__ int warp_offsets[7];
  __shared__ int batch_total;

  if (threadIdx.x == 0) {
    mbarrier_init(&load_barrier[0]);
    mbarrier_init(&load_barrier[1]);
    scan_ready[0] = 0;
    scan_ready[1] = 0;
  }
  for (std::size_t i = threadIdx.x; i < capacity; i += blockDim.x) {
    segment_out[i] = Entry{kEmpty, 0};
  }
  __syncthreads();

  const std::size_t tile_count = ceil_div(capacity, kTileEntries);
  if (threadIdx.x == 0) {
    const std::size_t preload = tile_count < 2 ? tile_count : 2;
    for (std::size_t tile = 0; tile < preload; ++tile) {
      const int stage = static_cast<int>(tile & 1);
      const std::size_t begin = tile * kTileEntries;
      input_entries[stage] = min(capacity - begin, kTileEntries);
      __threadfence_block();
      const int bytes =
          static_cast<int>(input_entries[stage] * sizeof(Entry));
      mbarrier_arrive_expect_tx(&load_barrier[stage], bytes);
      bulk_g2s(entries + stage * kTileEntries, segment_in + begin, bytes,
               &load_barrier[stage]);
    }

    for (std::size_t tile = 0; tile < tile_count; ++tile) {
      const int stage = static_cast<int>(tile & 1);
      while (atomicAdd(&scan_ready[stage], 0) <
             static_cast<int>(tile + 1)) {
        __nanosleep(64);
      }

      const std::size_t next_tile = tile + 2;
      if (next_tile < tile_count) {
        const std::size_t begin = next_tile * kTileEntries;
        input_entries[stage] = min(capacity - begin, kTileEntries);
        __threadfence_block();
        const int bytes =
            static_cast<int>(input_entries[stage] * sizeof(Entry));
        mbarrier_arrive_expect_tx(&load_barrier[stage], bytes);
        bulk_g2s(entries + stage * kTileEntries, segment_in + begin, bytes,
                 &load_barrier[stage]);
      }
    }
  } else if (threadIdx.x >= kProducerThreads) {
    const int consumer_id = threadIdx.x - kProducerThreads;
    const int lane = consumer_id & 31;
    const int consumer_warp = consumer_id >> 5;
    std::size_t live_base = 0;

    for (std::size_t tile = 0; tile < tile_count; ++tile) {
      const int stage = static_cast<int>(tile & 1);
      const int phase = static_cast<int>((tile / 2) & 1);
      mbarrier_wait(&load_barrier[stage], phase);
      Entry* stage_in = entries + stage * kTileEntries;
      const std::size_t in_count = input_entries[stage];

      if constexpr (strategy == 0) {
        for (std::size_t begin = 0; begin < in_count;
             begin += kConsumerThreads) {
          const std::size_t local = begin + consumer_id;
          const Entry entry =
              local < in_count ? stage_in[local] : Entry{kEmpty, 0};
          const bool valid = local < in_count && entry.key != kEmpty;
          const unsigned mask = __ballot_sync(0xffffffffu, valid);
          if (lane == 0) {
            warp_counts[consumer_warp] = __popc(mask);
          }
          consumer_barrier();

          if (consumer_warp == 0) {
            if (lane < 7) {
              int offset = 0;
              for (int i = 0; i < lane; ++i) {
                offset += warp_counts[i];
              }
              warp_offsets[lane] = offset;
            }
            if (lane == 0) {
              int total = 0;
              for (int i = 0; i < 7; ++i) {
                total += warp_counts[i];
              }
              batch_total = total;
            }
          }
          consumer_barrier();

          if (valid) {
            const unsigned lower_mask =
                lane == 0 ? 0u : ((1u << lane) - 1u);
            const std::size_t rank =
                live_base + warp_offsets[consumer_warp] +
                __popc(mask & lower_mask);
            const std::size_t destination =
                redistributed_position(rank, capacity, live_count);
            segment_out[destination] = entry;
          }
          consumer_barrier();
          live_base += batch_total;
        }
      } else {
        // Give each consumer a contiguous slice. A thread counts at most five
        // entries for a full 16 KiB tile, then one warp/CTA scan assigns the
        // stable base rank for the entire slice.
        const std::size_t chunk_begin =
            (std::size_t(consumer_id) * in_count) / kConsumerThreads;
        const std::size_t chunk_end =
            (std::size_t(consumer_id + 1) * in_count) / kConsumerThreads;
        int thread_live = 0;
        for (std::size_t local = chunk_begin; local < chunk_end; ++local) {
          thread_live += stage_in[local].key != kEmpty;
        }

        int inclusive = thread_live;
        for (int offset = 1; offset < 32; offset <<= 1) {
          const int other =
              __shfl_up_sync(0xffffffffu, inclusive, offset);
          if (lane >= offset) {
            inclusive += other;
          }
        }
        if (lane == 31) {
          warp_counts[consumer_warp] = inclusive;
        }
        consumer_barrier();

        if (consumer_warp == 0) {
          if (lane < 7) {
            int offset = 0;
            for (int i = 0; i < lane; ++i) {
              offset += warp_counts[i];
            }
            warp_offsets[lane] = offset;
          }
          if (lane == 0) {
            int total = 0;
            for (int i = 0; i < 7; ++i) {
              total += warp_counts[i];
            }
            batch_total = total;
          }
        }
        consumer_barrier();

        std::size_t rank =
            live_base + warp_offsets[consumer_warp] +
            inclusive - thread_live;
        for (std::size_t local = chunk_begin; local < chunk_end; ++local) {
          const Entry entry = stage_in[local];
          if (entry.key != kEmpty) {
            if constexpr (strategy == 1) {
              const std::size_t destination =
                  redistributed_position(rank, capacity, live_count);
              segment_out[destination] = entry;
            } else {
              compact[rank - live_base] = entry;
            }
            ++rank;
          }
        }
        consumer_barrier();
        if constexpr (strategy == 2) {
          for (std::size_t local = consumer_id; local < batch_total;
               local += kConsumerThreads) {
            const std::size_t rank = live_base + local;
            const std::size_t destination =
                redistributed_position(rank, capacity, live_count);
            segment_out[destination] = compact[local];
          }
          consumer_barrier();
        }
        live_base += batch_total;
      }

      if (threadIdx.x == kProducerThreads) {
        atomicExch(&scan_ready[stage], static_cast<int>(tile + 1));
      }
    }
  }
#endif
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    auto value = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--segment-bytes")) {
      options.segment_bytes = std::stoull(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--density")) {
      options.density = std::stod(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--working-set-mb")) {
      options.working_set_mb = std::stoull(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      options.warmup = std::stoi(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--iterations")) {
      options.iterations = std::stoi(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--mode")) {
      options.mode = value(argv[i]);
    } else if (!std::strcmp(argv[i], "--layout")) {
      options.layout = value(argv[i]);
    } else if (!std::strcmp(argv[i], "--help")) {
      std::puts("segment_bench [--segment-bytes N] [--density F] "
                "[--working-set-mb N] [--warmup N] [--iterations N] "
                "[--layout prefix|spread] "
                "[--mode all|scalar|cp_async|tma|tma_tiled|tma_pipeline|"
                "gap_all|gap_scan|gap_tma_pipeline|gap_tma_chunked|"
                "gap_tma_buffered]");
      std::exit(0);
    } else {
      throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
    }
  }
  if (options.segment_bytes < 256 || options.segment_bytes % sizeof(Entry)) {
    throw std::runtime_error("segment bytes must be >=256 and divisible by 16");
  }
  if (!(options.density > 0.0 && options.density <= 1.0)) {
    throw std::runtime_error("density must be in (0, 1]");
  }
  if (options.iterations <= 0 || options.warmup < 0) {
    throw std::runtime_error("invalid iteration count");
  }
  if (options.layout != "prefix" && options.layout != "spread") {
    throw std::runtime_error("layout must be prefix or spread");
  }
  const bool gap_mode = options.mode == "gap_all" ||
                        options.mode == "gap_scan" ||
                        options.mode == "gap_tma_pipeline" ||
                        options.mode == "gap_tma_chunked" ||
                        options.mode == "gap_tma_buffered";
  if (options.layout != "prefix" && !gap_mode) {
    throw std::runtime_error(
        "non-prefix layouts require gap_all, gap_scan, or gap_tma_pipeline");
  }
  return options;
}

template <Mode mode>
void configure_smem(std::size_t bytes) {
  CUDA_CHECK(cudaFuncSetAttribute(redistribute_kernel<mode>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(bytes)));
}

template <Mode mode>
float time_mode(const Entry* input, Entry* output, int segments,
                std::size_t capacity, std::size_t live_count,
                std::size_t smem_bytes, int warmup, int iterations) {
  configure_smem<mode>(smem_bytes);
  for (int i = 0; i < warmup; ++i) {
    redistribute_kernel<mode><<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    redistribute_kernel<mode><<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / iterations;
}

float time_tma_tiled_mode(const Entry* input, Entry* output, int segments,
                          std::size_t capacity, std::size_t live_count,
                          int warmup, int iterations) {
  constexpr std::size_t smem_bytes = 2 * kMaxBulkBytes;
  CUDA_CHECK(cudaFuncSetAttribute(
      redistribute_tma_tiled_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  for (int i = 0; i < warmup; ++i) {
    redistribute_tma_tiled_kernel<<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    redistribute_tma_tiled_kernel<<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / iterations;
}

float time_tma_pipeline_mode(const Entry* input, Entry* output, int segments,
                             std::size_t capacity, std::size_t live_count,
                             int warmup, int iterations) {
  constexpr std::size_t smem_bytes = 4 * kMaxBulkBytes;
  CUDA_CHECK(cudaFuncSetAttribute(
      redistribute_tma_pipeline_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  for (int i = 0; i < warmup; ++i) {
    redistribute_tma_pipeline_kernel<<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    redistribute_tma_pipeline_kernel<<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / iterations;
}

float time_gap_scan_mode(const Entry* input, Entry* output, int segments,
                         std::size_t capacity, std::size_t live_count,
                         int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    redistribute_gap_scan_kernel<<<segments, kThreads>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    redistribute_gap_scan_kernel<<<segments, kThreads>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / iterations;
}

template <int strategy>
float time_gap_tma_pipeline_mode(const Entry* input, Entry* output,
                                 int segments, std::size_t capacity,
                                 std::size_t live_count, int warmup,
                                 int iterations) {
  constexpr std::size_t smem_bytes =
      (strategy == 2 ? 3 : 2) * kMaxBulkBytes;
  CUDA_CHECK(cudaFuncSetAttribute(
      redistribute_gap_tma_pipeline_kernel<strategy>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  for (int i = 0; i < warmup; ++i) {
    redistribute_gap_tma_pipeline_kernel<strategy>
        <<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    redistribute_gap_tma_pipeline_kernel<strategy>
        <<<segments, kThreads, smem_bytes>>>(
        input, output, capacity, live_count);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / iterations;
}

bool validate_first_segment(const Entry* output, std::size_t capacity,
                            std::size_t live_count) {
  std::vector<Entry> host(capacity);
  CUDA_CHECK(cudaMemcpy(host.data(), output, capacity * sizeof(Entry),
                        cudaMemcpyDeviceToHost));
  std::vector<bool> occupied(capacity, false);
  for (std::size_t i = 0; i < live_count; ++i) {
    const std::size_t dst = redistributed_position(i, capacity, live_count);
    occupied[dst] = true;
    if (host[dst].key != i + 1 ||
        host[dst].value != ((i + 1) ^ 0x5a5a5a5aULL)) {
      return false;
    }
  }
  for (std::size_t i = 0; i < capacity; ++i) {
    if (!occupied[i] && host[i].key != kEmpty) {
      return false;
    }
  }
  return true;
}

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kScalar:
      return "scalar";
    case Mode::kCpAsync:
      return "cp_async";
    case Mode::kTma:
      return "tma_bulk";
    case Mode::kTmaTiled:
      return "tma_tiled";
    case Mode::kTmaPipeline:
      return "tma_pipeline";
    case Mode::kGapScan:
      return "gap_scan";
    case Mode::kGapTmaPipeline:
      return "gap_tma_pipeline";
    case Mode::kGapTmaChunked:
      return "gap_tma_chunked";
    case Mode::kGapTmaBuffered:
      return "gap_tma_buffered";
  }
  return "unknown";
}

bool requested(const Options& options, Mode mode) {
  const bool gap = mode == Mode::kGapScan ||
                   mode == Mode::kGapTmaPipeline ||
                   mode == Mode::kGapTmaChunked ||
                   mode == Mode::kGapTmaBuffered;
  return (options.mode == "all" && !gap) ||
         (options.mode == "gap_all" && gap) ||
         options.mode == mode_name(mode) ||
         (mode == Mode::kTma && options.mode == "tma");
}

Layout input_layout(const Options& options) {
  return options.layout == "spread" ? Layout::kSpread : Layout::kPrefix;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::size_t capacity = options.segment_bytes / sizeof(Entry);
    const std::size_t live_count = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::floor(capacity * options.density)));
    const std::size_t working_bytes = options.working_set_mb * 1024ULL * 1024ULL;
    const int segments = static_cast<int>(
        std::max<std::size_t>(1, working_bytes / options.segment_bytes));
    const std::size_t total_entries = std::size_t(segments) * capacity;
    const std::size_t smem_bytes = 2 * options.segment_bytes;

    const bool needs_full_segment_smem =
        options.mode == "all" || options.mode == "scalar" ||
        options.mode == "cp_async" || options.mode == "tma" ||
        options.mode == "tma_bulk";
    if (needs_full_segment_smem &&
        smem_bytes > prop.sharedMemPerBlockOptin) {
      throw std::runtime_error("requested two-buffer shared memory exceeds "
                               "device opt-in limit");
    }

    Entry* input = nullptr;
    Entry* output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, total_entries * sizeof(Entry)));
    CUDA_CHECK(cudaMalloc(&output, total_entries * sizeof(Entry)));
    initialize_input<<<std::min<int>(65535, (total_entries + 255) / 256), 256>>>(
        input, output, total_entries, capacity, live_count,
        input_layout(options));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::puts("device,cc,mode,layout,segment_bytes,density,segments,"
              "live_entries,latency_ms,effective_gbps,valid");
    const Mode modes[] = {Mode::kScalar, Mode::kCpAsync, Mode::kTma,
                          Mode::kTmaTiled, Mode::kTmaPipeline,
                          Mode::kGapScan, Mode::kGapTmaPipeline,
                          Mode::kGapTmaChunked, Mode::kGapTmaBuffered};
    for (Mode mode : modes) {
      if (!requested(options, mode)) {
        continue;
      }
      if (mode == Mode::kCpAsync && prop.major < 8) {
        continue;
      }
      if ((mode == Mode::kTma || mode == Mode::kTmaTiled ||
           mode == Mode::kTmaPipeline ||
           mode == Mode::kGapTmaPipeline ||
           mode == Mode::kGapTmaChunked ||
           mode == Mode::kGapTmaBuffered) &&
          prop.major < 9) {
        if (options.mode != "all") {
          throw std::runtime_error("TMA mode requires compute capability 9.x");
        }
        continue;
      }

      float latency_ms = 0.0f;
      if (mode == Mode::kScalar) {
        latency_ms = time_mode<Mode::kScalar>(
            input, output, segments, capacity, live_count, smem_bytes,
            options.warmup, options.iterations);
      } else if (mode == Mode::kCpAsync) {
        latency_ms = time_mode<Mode::kCpAsync>(
            input, output, segments, capacity, live_count, smem_bytes,
            options.warmup, options.iterations);
      } else if (mode == Mode::kTma) {
        latency_ms = time_mode<Mode::kTma>(
            input, output, segments, capacity, live_count, smem_bytes,
            options.warmup, options.iterations);
      } else if (mode == Mode::kTmaTiled) {
        latency_ms = time_tma_tiled_mode(
            input, output, segments, capacity, live_count, options.warmup,
            options.iterations);
      } else if (mode == Mode::kTmaPipeline) {
        latency_ms = time_tma_pipeline_mode(
            input, output, segments, capacity, live_count, options.warmup,
            options.iterations);
      } else if (mode == Mode::kGapScan) {
        latency_ms = time_gap_scan_mode(
            input, output, segments, capacity, live_count, options.warmup,
            options.iterations);
      } else if (mode == Mode::kGapTmaPipeline) {
        latency_ms = time_gap_tma_pipeline_mode<0>(
            input, output, segments, capacity, live_count, options.warmup,
            options.iterations);
      } else if (mode == Mode::kGapTmaChunked) {
        latency_ms = time_gap_tma_pipeline_mode<1>(
            input, output, segments, capacity, live_count, options.warmup,
            options.iterations);
      } else {
        latency_ms = time_gap_tma_pipeline_mode<2>(
            input, output, segments, capacity, live_count, options.warmup,
            options.iterations);
      }
      const bool valid = validate_first_segment(output, capacity, live_count);
      const double bytes = 2.0 * segments * options.segment_bytes;
      const double gbps = bytes / (latency_ms * 1.0e6);
      std::printf("\"%s\",%d.%d,%s,%s,%zu,%.4f,%d,%zu,%.6f,%.3f,%d\n",
                  prop.name, prop.major, prop.minor, mode_name(mode),
                  options.layout.c_str(), options.segment_bytes,
                  options.density, segments, live_count, latency_ms, gbps,
                  valid ? 1 : 0);
      if (!valid) {
        throw std::runtime_error(std::string("validation failed for ") +
                                 mode_name(mode));
      }
    }

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(input));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "segment_bench: %s\n", error.what());
    return 1;
  }
}
