#include <cuda_runtime.h>

#include <cub/block/block_scan.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct alignas(16) Entry {
  std::uint64_t key;
  std::uint64_t value;
};

struct Update {
  unsigned int src;
  unsigned int dst;
  unsigned int sequence;
  unsigned int op;
  std::uint64_t value;
};

static_assert(sizeof(Entry) == 16, "Entry must match the PMA benchmark");

constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
constexpr unsigned int kDelete = 0;
constexpr unsigned int kInsert = 1;
constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(call) check_cuda((call), #call)

struct Options {
  unsigned int vertices = 4096;
  unsigned int segment_capacity = 256;
  double density = 0.5;
  std::size_t batch_size = 65536;
  double insert_ratio = 0.5;
  double duplicate_ratio = 0.1;
  std::string distribution = "uniform";
  std::string mode = "adaptive";
  unsigned int parallel_threshold = 1024;
  double zipf_skew = 1.1;
  std::uint64_t seed = 7;
  int warmup = 3;
  int iterations = 10;
};

struct UpdateOrder {
  __host__ __device__ bool operator()(const Update& a,
                                      const Update& b) const {
    if (a.src != b.src) {
      return a.src < b.src;
    }
    if (a.dst != b.dst) {
      return a.dst < b.dst;
    }
    return a.sequence > b.sequence;
  }
};

struct SameEdge {
  __host__ __device__ bool operator()(const Update& a,
                                      const Update& b) const {
    return a.src == b.src && a.dst == b.dst;
  }
};

__host__ __device__ __forceinline__ std::size_t redistributed_position(
    std::size_t live_index, std::size_t capacity, std::size_t live_count) {
  return (live_index * capacity) / live_count;
}

std::uint64_t base_value(unsigned int src, unsigned int dst) {
  return (std::uint64_t{src} << 32) ^ dst ^ 0x5a5a5a5aULL;
}

std::uint64_t update_value(unsigned int src, unsigned int dst,
                           unsigned int sequence) {
  return (std::uint64_t{sequence} << 32) ^
         (std::uint64_t{src} * 0x9e3779b97f4a7c15ULL) ^ dst;
}

std::vector<Entry> make_initial_graph(const Options& options,
                                      unsigned int live_count) {
  const std::size_t total =
      std::size_t{options.vertices} * options.segment_capacity;
  std::vector<Entry> graph(total, Entry{kEmpty, 0});
  for (unsigned int src = 0; src < options.vertices; ++src) {
    Entry* segment =
        graph.data() + std::size_t{src} * options.segment_capacity;
    for (unsigned int i = 0; i < live_count; ++i) {
      const unsigned int dst = 2 * i;
      const std::size_t position = redistributed_position(
          i, options.segment_capacity, live_count);
      segment[position] = Entry{dst, base_value(src, dst)};
    }
  }
  return graph;
}

std::vector<Update> make_updates(const Options& options) {
  if (options.batch_size >
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error("batch size exceeds 32-bit sequence range");
  }

  std::mt19937_64 rng(options.seed);
  std::uniform_int_distribution<unsigned int> uniform_src(
      0, options.vertices - 1);
  std::uniform_int_distribution<unsigned int> dst_distribution(
      0, 4 * options.segment_capacity - 1);
  std::bernoulli_distribution insert_distribution(options.insert_ratio);
  std::bernoulli_distribution duplicate_distribution(options.duplicate_ratio);

  std::vector<double> zipf_weights;
  std::discrete_distribution<unsigned int> zipf_src;
  if (options.distribution == "zipf") {
    zipf_weights.resize(options.vertices);
    for (unsigned int i = 0; i < options.vertices; ++i) {
      zipf_weights[i] =
          1.0 / std::pow(static_cast<double>(i + 1), options.zipf_skew);
    }
    zipf_src = std::discrete_distribution<unsigned int>(
        zipf_weights.begin(), zipf_weights.end());
  }

  std::vector<Update> updates;
  updates.reserve(options.batch_size);
  for (std::size_t i = 0; i < options.batch_size; ++i) {
    unsigned int src = options.distribution == "zipf" ? zipf_src(rng)
                                                        : uniform_src(rng);
    unsigned int dst = dst_distribution(rng);
    if (i && duplicate_distribution(rng)) {
      std::uniform_int_distribution<std::size_t> prior(0, i - 1);
      const Update& duplicate = updates[prior(rng)];
      src = duplicate.src;
      dst = duplicate.dst;
    }
    const unsigned int op =
        insert_distribution(rng) ? kInsert : kDelete;
    const unsigned int sequence = static_cast<unsigned int>(i);
    updates.push_back(
        Update{src, dst, sequence, op,
               op == kInsert ? update_value(src, dst, sequence) : 0});
  }
  return updates;
}

std::vector<std::vector<Entry>> make_reference(
    const Options& options, unsigned int initial_live,
    const std::vector<Update>& updates) {
  std::unordered_map<std::uint64_t, Update> last_update;
  last_update.reserve(updates.size());
  for (const Update& update : updates) {
    const std::uint64_t key =
        (std::uint64_t{update.src} << 32) | update.dst;
    last_update[key] = update;
  }

  std::vector<std::vector<Update>> by_source(options.vertices);
  for (const auto& item : last_update) {
    by_source[item.second.src].push_back(item.second);
  }
  for (auto& source_updates : by_source) {
    std::sort(source_updates.begin(), source_updates.end(),
              [](const Update& a, const Update& b) {
                return a.dst < b.dst;
              });
  }

  std::vector<std::vector<Entry>> reference(options.vertices);
  for (unsigned int src = 0; src < options.vertices; ++src) {
    auto& expected = reference[src];
    const auto& source_updates = by_source[src];
    unsigned int base_index = 0;
    std::size_t update_index = 0;
    while (base_index < initial_live ||
           update_index < source_updates.size()) {
      const std::uint64_t base_key =
          base_index < initial_live ? 2ULL * base_index : kEmpty;
      const std::uint64_t update_key =
          update_index < source_updates.size()
              ? source_updates[update_index].dst
              : kEmpty;
      if (base_key < update_key) {
        expected.push_back(
            Entry{base_key, base_value(src, static_cast<unsigned int>(base_key))});
        ++base_index;
      } else if (update_key < base_key) {
        const Update& update = source_updates[update_index++];
        if (update.op == kInsert) {
          expected.push_back(Entry{update.dst, update.value});
        }
      } else {
        const Update& update = source_updates[update_index++];
        ++base_index;
        if (update.op == kInsert) {
          expected.push_back(Entry{update.dst, update.value});
        }
      }
    }
    if (expected.size() > options.segment_capacity) {
      throw std::runtime_error(
          "reference graph exceeds segment capacity; reduce batch or skew");
    }
  }
  return reference;
}

__global__ void count_updates_kernel(const Update* updates,
                                     std::size_t update_count,
                                     unsigned int* segment_counts) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < update_count;
       i += std::size_t{gridDim.x} * blockDim.x) {
    atomicAdd(segment_counts + updates[i].src, 1U);
  }
}

__global__ void plan_segments_kernel(
    const unsigned int* segment_counts, unsigned int vertices,
    unsigned int capacity, unsigned int parallel_threshold, int mode,
    unsigned int* serial_segments, unsigned int* parallel_segments,
    unsigned int* affected_count, unsigned int* serial_count,
    unsigned int* parallel_count) {
  for (unsigned int src = blockIdx.x * blockDim.x + threadIdx.x;
       src < vertices; src += gridDim.x * blockDim.x) {
    if (segment_counts[src] != 0) {
      atomicAdd(affected_count, 1U);
      const bool use_parallel =
          mode == 1 ||
          (mode == 2 &&
           std::uint64_t{capacity} + segment_counts[src] >=
               parallel_threshold);
      if (use_parallel) {
        const unsigned int slot = atomicAdd(parallel_count, 1U);
        parallel_segments[slot] = src;
      } else {
        const unsigned int slot = atomicAdd(serial_count, 1U);
        serial_segments[slot] = src;
      }
    }
  }
}

__device__ __forceinline__ void emit_entry(
    Entry* scratch_segment, unsigned int capacity, unsigned int* output_count,
    unsigned int* overflow, Entry entry) {
  if (*output_count < capacity) {
    scratch_segment[*output_count] = entry;
  } else {
    *overflow = 1;
  }
  ++*output_count;
}

__global__ void merge_one_level_kernel(
    const Entry* base, Entry* output, Entry* scratch,
    unsigned int* live_counts, const Update* updates,
    const unsigned int* update_offsets,
    const unsigned int* affected_segments, unsigned int capacity,
    unsigned int* overflow) {
  const unsigned int src = affected_segments[blockIdx.x];
  const Entry* base_segment = base + std::size_t{src} * capacity;
  Entry* output_segment = output + std::size_t{src} * capacity;
  Entry* scratch_segment = scratch + std::size_t{src} * capacity;
  __shared__ unsigned int result_count;
  __shared__ unsigned int segment_overflow;

  if (threadIdx.x == 0) {
    unsigned int base_index = 0;
    unsigned int update_index = update_offsets[src];
    const unsigned int update_end = update_offsets[src + 1];
    unsigned int emitted = 0;
    unsigned int did_overflow = 0;

    while (base_index < capacity || update_index < update_end) {
      while (base_index < capacity &&
             base_segment[base_index].key == kEmpty) {
        ++base_index;
      }
      const std::uint64_t base_key =
          base_index < capacity ? base_segment[base_index].key : kEmpty;
      const std::uint64_t update_key =
          update_index < update_end ? updates[update_index].dst : kEmpty;
      if (base_key == kEmpty && update_key == kEmpty) {
        break;
      }

      if (base_key < update_key) {
        emit_entry(scratch_segment, capacity, &emitted, &did_overflow,
                   base_segment[base_index++]);
      } else if (update_key < base_key) {
        const Update update = updates[update_index++];
        if (update.op == kInsert) {
          emit_entry(scratch_segment, capacity, &emitted, &did_overflow,
                     Entry{update.dst, update.value});
        }
      } else {
        const Update update = updates[update_index++];
        ++base_index;
        if (update.op == kInsert) {
          emit_entry(scratch_segment, capacity, &emitted, &did_overflow,
                     Entry{update.dst, update.value});
        }
      }
    }
    result_count = min(emitted, capacity);
    segment_overflow = did_overflow;
    overflow[src] = did_overflow;
    live_counts[src] = result_count;
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < capacity; i += blockDim.x) {
    output_segment[i] = Entry{kEmpty, 0};
  }
  __syncthreads();

  if (!segment_overflow && result_count != 0) {
    for (unsigned int i = threadIdx.x; i < result_count;
         i += blockDim.x) {
      const std::size_t destination =
          redistributed_position(i, capacity, result_count);
      output_segment[destination] = scratch_segment[i];
    }
  }
}

__device__ __forceinline__ unsigned int lower_bound_entries(
    const Entry* entries, unsigned int count, std::uint64_t key) {
  unsigned int first = 0;
  while (first < count) {
    const unsigned int middle = first + (count - first) / 2;
    if (entries[middle].key < key) {
      first = middle + 1;
    } else {
      count = middle;
    }
  }
  return first;
}

__device__ __forceinline__ unsigned int lower_bound_updates(
    const Update* updates, unsigned int first, unsigned int last,
    std::uint64_t key) {
  while (first < last) {
    const unsigned int middle = first + (last - first) / 2;
    if (updates[middle].dst < key) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return first;
}

__global__ void prepare_parallel_merge_kernel(
    const Entry* base, Entry* dense_base, unsigned int* live_counts,
    unsigned int* base_counts, const Update* updates,
    const unsigned int* update_offsets,
    const unsigned int* parallel_segments, unsigned int capacity,
    int* update_prefix, int* segment_effects, unsigned int* overflow) {
  using BlockScan = cub::BlockScan<int, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int carry;
  __shared__ unsigned int dense_count;

  const unsigned int src = parallel_segments[blockIdx.x];
  const Entry* base_segment = base + std::size_t{src} * capacity;
  Entry* dense_segment = dense_base + std::size_t{src} * capacity;

  if (threadIdx.x == 0) {
    carry = 0;
  }
  __syncthreads();
  for (unsigned int tile = 0; tile < capacity; tile += blockDim.x) {
    const unsigned int index = tile + threadIdx.x;
    const int live =
        index < capacity && base_segment[index].key != kEmpty ? 1 : 0;
    int prefix = 0;
    int tile_total = 0;
    const int tile_base = carry;
    BlockScan(scan_storage).ExclusiveSum(live, prefix, tile_total);
    if (live) {
      dense_segment[tile_base + prefix] = base_segment[index];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      carry += tile_total;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dense_count = static_cast<unsigned int>(carry);
    base_counts[src] = dense_count;
    carry = 0;
  }
  __syncthreads();

  const unsigned int update_begin = update_offsets[src];
  const unsigned int update_end = update_offsets[src + 1];
  for (unsigned int tile = update_begin; tile < update_end;
       tile += blockDim.x) {
    const unsigned int update_index = tile + threadIdx.x;
    int effect = 0;
    if (update_index < update_end) {
      const Update update = updates[update_index];
      const unsigned int base_position = lower_bound_entries(
          dense_segment, dense_count, update.dst);
      const bool present =
          base_position < dense_count &&
          dense_segment[base_position].key == update.dst;
      effect = update.op == kInsert ? (present ? 0 : 1)
                                    : (present ? -1 : 0);
    }
    int prefix = 0;
    int tile_total = 0;
    const int tile_base = carry;
    BlockScan(scan_storage).ExclusiveSum(effect, prefix, tile_total);
    if (update_index < update_end) {
      update_prefix[update_index] = tile_base + prefix;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      carry += tile_total;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    segment_effects[src] = carry;
    const int result_count = static_cast<int>(dense_count) + carry;
    const bool did_overflow =
        result_count < 0 || result_count > static_cast<int>(capacity);
    overflow[src] = did_overflow ? 1U : 0U;
    live_counts[src] = did_overflow
                           ? 0U
                           : static_cast<unsigned int>(result_count);
  }
}

__global__ void scatter_parallel_merge_kernel(
    const Entry* dense_base, Entry* output, const unsigned int* live_counts,
    const unsigned int* base_counts, const Update* updates,
    const unsigned int* update_offsets,
    const unsigned int* parallel_segments, unsigned int capacity,
    const int* update_prefix, const int* segment_effects,
    const unsigned int* overflow) {
  const unsigned int src = parallel_segments[blockIdx.x];
  Entry* output_segment = output + std::size_t{src} * capacity;
  const Entry* dense_segment =
      dense_base + std::size_t{src} * capacity;
  const unsigned int result_count = live_counts[src];
  const unsigned int dense_count = base_counts[src];
  const unsigned int update_begin = update_offsets[src];
  const unsigned int update_end = update_offsets[src + 1];

  for (unsigned int i = threadIdx.x; i < capacity; i += blockDim.x) {
    output_segment[i] = Entry{kEmpty, 0};
  }
  __syncthreads();
  if (overflow[src] != 0 || result_count == 0) {
    return;
  }

  for (unsigned int base_index = threadIdx.x; base_index < dense_count;
       base_index += blockDim.x) {
    const Entry entry = dense_segment[base_index];
    const unsigned int update_position = lower_bound_updates(
        updates, update_begin, update_end, entry.key);
    const bool overridden =
        update_position < update_end &&
        updates[update_position].dst == entry.key;
    if (!overridden) {
      const int delta_before =
          update_position < update_end
              ? update_prefix[update_position]
              : segment_effects[src];
      const int rank = static_cast<int>(base_index) + delta_before;
      const std::size_t destination = redistributed_position(
          static_cast<unsigned int>(rank), capacity, result_count);
      output_segment[destination] = entry;
    }
  }

  for (unsigned int update_index = update_begin + threadIdx.x;
       update_index < update_end; update_index += blockDim.x) {
    const Update update = updates[update_index];
    if (update.op != kInsert) {
      continue;
    }
    const unsigned int base_position = lower_bound_entries(
        dense_segment, dense_count, update.dst);
    const int rank =
        static_cast<int>(base_position) + update_prefix[update_index];
    const std::size_t destination = redistributed_position(
        static_cast<unsigned int>(rank), capacity, result_count);
    output_segment[destination] = Entry{update.dst, update.value};
  }
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  return milliseconds;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    auto value = [&](const char* flag) {
      if (++i >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return argv[i];
    };
    if (!std::strcmp(argv[i], "--vertices")) {
      options.vertices = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--segment-capacity")) {
      options.segment_capacity = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--density")) {
      options.density = std::stod(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--batch-size")) {
      options.batch_size = std::stoull(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--insert-ratio")) {
      options.insert_ratio = std::stod(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--duplicate-ratio")) {
      options.duplicate_ratio = std::stod(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--distribution")) {
      options.distribution = value(argv[i]);
    } else if (!std::strcmp(argv[i], "--mode")) {
      options.mode = value(argv[i]);
    } else if (!std::strcmp(argv[i], "--parallel-threshold")) {
      options.parallel_threshold = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--zipf-skew")) {
      options.zipf_skew = std::stod(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--seed")) {
      options.seed = std::stoull(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      options.warmup = std::stoi(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--iterations")) {
      options.iterations = std::stoi(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--help")) {
      std::puts(
          "update_bench [--vertices N] [--segment-capacity N] "
          "[--density F] [--batch-size N] [--insert-ratio F] "
          "[--duplicate-ratio F] [--distribution uniform|zipf] "
          "[--mode serial|parallel|adaptive] [--parallel-threshold N] "
          "[--zipf-skew F] [--seed N] [--warmup N] [--iterations N]");
      std::exit(0);
    } else {
      throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
    }
  }

  if (options.vertices == 0 || options.segment_capacity == 0 ||
      options.batch_size == 0) {
    throw std::runtime_error("vertices, capacity, and batch must be nonzero");
  }
  if (options.segment_capacity >
      std::numeric_limits<unsigned int>::max() / 4) {
    throw std::runtime_error("segment capacity is too large");
  }
  if (!(options.density > 0.0 && options.density < 1.0) ||
      !(options.insert_ratio >= 0.0 && options.insert_ratio <= 1.0) ||
      !(options.duplicate_ratio >= 0.0 && options.duplicate_ratio < 1.0)) {
    throw std::runtime_error("invalid density or update ratio");
  }
  if (options.distribution != "uniform" &&
      options.distribution != "zipf") {
    throw std::runtime_error("distribution must be uniform or zipf");
  }
  if (options.mode != "serial" && options.mode != "parallel" &&
      options.mode != "adaptive") {
    throw std::runtime_error("mode must be serial, parallel, or adaptive");
  }
  if (options.parallel_threshold == 0) {
    throw std::runtime_error("parallel threshold must be nonzero");
  }
  if (options.zipf_skew <= 0.0 || options.warmup < 0 ||
      options.iterations <= 0) {
    throw std::runtime_error("invalid skew or iteration count");
  }
  return options;
}

bool validate_output(const Options& options,
                     const std::vector<std::vector<Entry>>& reference,
                     const std::vector<Entry>& output,
                     const std::vector<unsigned int>& live_counts,
                     const std::vector<unsigned int>& overflow) {
  for (unsigned int src = 0; src < options.vertices; ++src) {
    if (overflow[src] != 0 ||
        live_counts[src] != reference[src].size()) {
      return false;
    }
    const Entry* segment =
        output.data() + std::size_t{src} * options.segment_capacity;
    for (unsigned int i = 0; i < live_counts[src]; ++i) {
      const std::size_t position = redistributed_position(
          i, options.segment_capacity, live_counts[src]);
      if (segment[position].key != reference[src][i].key ||
          segment[position].value != reference[src][i].value) {
        return false;
      }
    }
    for (unsigned int position = 0;
         position < options.segment_capacity; ++position) {
      const Entry& entry = segment[position];
      if (entry.key == kEmpty) {
        continue;
      }
      const std::size_t rank = std::lower_bound(
          reference[src].begin(), reference[src].end(), entry.key,
          [](const Entry& candidate, std::uint64_t key) {
            return candidate.key < key;
          }) - reference[src].begin();
      if (rank >= reference[src].size() ||
          redistributed_position(rank, options.segment_capacity,
                                 reference[src].size()) != position) {
        return false;
      }
    }
  }
  return true;
}

int run_benchmark(const Options& options) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  const unsigned int initial_live = std::max(
      1U, static_cast<unsigned int>(
              std::floor(options.segment_capacity * options.density)));
  const std::size_t total_entries =
      std::size_t{options.vertices} * options.segment_capacity;
  if (total_entries >
      std::numeric_limits<std::size_t>::max() / sizeof(Entry)) {
    throw std::runtime_error("graph allocation size overflows size_t");
  }

  const std::vector<Entry> initial =
      make_initial_graph(options, initial_live);
  const std::vector<Update> updates = make_updates(options);
  const std::vector<std::vector<Entry>> reference =
      make_reference(options, initial_live, updates);

  thrust::device_vector<Entry> d_base(initial.begin(), initial.end());
  thrust::device_vector<Entry> d_output(total_entries);
  thrust::device_vector<Entry> d_scratch(total_entries);
  thrust::device_vector<Update> d_updates(updates.begin(), updates.end());
  thrust::device_vector<unsigned int> d_segment_counts(
      options.vertices + 1, 0);
  thrust::device_vector<unsigned int> d_update_offsets(
      options.vertices + 1, 0);
  thrust::device_vector<unsigned int> d_serial_segments(options.vertices);
  thrust::device_vector<unsigned int> d_parallel_segments(options.vertices);
  thrust::device_vector<unsigned int> d_affected_count(1, 0);
  thrust::device_vector<unsigned int> d_serial_count(1, 0);
  thrust::device_vector<unsigned int> d_parallel_count(1, 0);
  thrust::device_vector<unsigned int> d_live_counts(
      options.vertices, initial_live);
  thrust::device_vector<unsigned int> d_base_counts(options.vertices, 0);
  thrust::device_vector<unsigned int> d_overflow(options.vertices, 0);
  thrust::device_vector<int> d_update_prefix(updates.size(), 0);
  thrust::device_vector<int> d_segment_effects(options.vertices, 0);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  thrust::sort(thrust::device, d_updates.begin(), d_updates.end(),
               UpdateOrder{});
  const auto unique_end = thrust::unique(
      thrust::device, d_updates.begin(), d_updates.end(), SameEdge{});
  const std::size_t unique_updates =
      static_cast<std::size_t>(unique_end - d_updates.begin());
  const int update_blocks = static_cast<int>(std::min<std::size_t>(
      65535, (unique_updates + kThreads - 1) / kThreads));
  count_updates_kernel<<<std::max(1, update_blocks), kThreads>>>(
      thrust::raw_pointer_cast(d_updates.data()), unique_updates,
      thrust::raw_pointer_cast(d_segment_counts.data()));
  thrust::exclusive_scan(thrust::device, d_segment_counts.begin(),
                         d_segment_counts.end(), d_update_offsets.begin());

  const int vertex_blocks = static_cast<int>(std::min<std::size_t>(
      65535, (std::size_t{options.vertices} + kThreads - 1) / kThreads));
  const int path_mode =
      options.mode == "serial" ? 0 : (options.mode == "parallel" ? 1 : 2);
  plan_segments_kernel<<<std::max(1, vertex_blocks), kThreads>>>(
      thrust::raw_pointer_cast(d_segment_counts.data()), options.vertices,
      options.segment_capacity, options.parallel_threshold, path_mode,
      thrust::raw_pointer_cast(d_serial_segments.data()),
      thrust::raw_pointer_cast(d_parallel_segments.data()),
      thrust::raw_pointer_cast(d_affected_count.data()),
      thrust::raw_pointer_cast(d_serial_count.data()),
      thrust::raw_pointer_cast(d_parallel_count.data()));
  CUDA_CHECK(cudaGetLastError());
  const float preprocess_ms = elapsed_ms(start, stop);

  unsigned int affected_count = 0;
  unsigned int serial_count = 0;
  unsigned int parallel_count = 0;
  CUDA_CHECK(cudaMemcpy(
      &affected_count, thrust::raw_pointer_cast(d_affected_count.data()),
      sizeof(affected_count), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      &serial_count, thrust::raw_pointer_cast(d_serial_count.data()),
      sizeof(serial_count), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      &parallel_count, thrust::raw_pointer_cast(d_parallel_count.data()),
      sizeof(parallel_count), cudaMemcpyDeviceToHost));
  if (affected_count == 0) {
    throw std::runtime_error("generated batch contains no affected segment");
  }
  if (serial_count + parallel_count != affected_count) {
    throw std::runtime_error("planner bucket counts are inconsistent");
  }

  CUDA_CHECK(cudaMemcpy(
      thrust::raw_pointer_cast(d_output.data()),
      thrust::raw_pointer_cast(d_base.data()),
      total_entries * sizeof(Entry), cudaMemcpyDeviceToDevice));

  auto launch_update = [&]() {
    if (serial_count != 0) {
      merge_one_level_kernel<<<serial_count, kThreads>>>(
          thrust::raw_pointer_cast(d_base.data()),
          thrust::raw_pointer_cast(d_output.data()),
          thrust::raw_pointer_cast(d_scratch.data()),
          thrust::raw_pointer_cast(d_live_counts.data()),
          thrust::raw_pointer_cast(d_updates.data()),
          thrust::raw_pointer_cast(d_update_offsets.data()),
          thrust::raw_pointer_cast(d_serial_segments.data()),
          options.segment_capacity,
          thrust::raw_pointer_cast(d_overflow.data()));
    }
    if (parallel_count != 0) {
      prepare_parallel_merge_kernel<<<parallel_count, kThreads>>>(
          thrust::raw_pointer_cast(d_base.data()),
          thrust::raw_pointer_cast(d_scratch.data()),
          thrust::raw_pointer_cast(d_live_counts.data()),
          thrust::raw_pointer_cast(d_base_counts.data()),
          thrust::raw_pointer_cast(d_updates.data()),
          thrust::raw_pointer_cast(d_update_offsets.data()),
          thrust::raw_pointer_cast(d_parallel_segments.data()),
          options.segment_capacity,
          thrust::raw_pointer_cast(d_update_prefix.data()),
          thrust::raw_pointer_cast(d_segment_effects.data()),
          thrust::raw_pointer_cast(d_overflow.data()));
      scatter_parallel_merge_kernel<<<parallel_count, kThreads>>>(
          thrust::raw_pointer_cast(d_scratch.data()),
          thrust::raw_pointer_cast(d_output.data()),
          thrust::raw_pointer_cast(d_live_counts.data()),
          thrust::raw_pointer_cast(d_base_counts.data()),
          thrust::raw_pointer_cast(d_updates.data()),
          thrust::raw_pointer_cast(d_update_offsets.data()),
          thrust::raw_pointer_cast(d_parallel_segments.data()),
          options.segment_capacity,
          thrust::raw_pointer_cast(d_update_prefix.data()),
          thrust::raw_pointer_cast(d_segment_effects.data()),
          thrust::raw_pointer_cast(d_overflow.data()));
    }
  };

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    launch_update();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    launch_update();
  }
  CUDA_CHECK(cudaGetLastError());
  const float update_ms =
      elapsed_ms(start, stop) / static_cast<float>(options.iterations);

  std::vector<Entry> output(total_entries);
  std::vector<unsigned int> live_counts(options.vertices);
  std::vector<unsigned int> overflow(options.vertices);
  CUDA_CHECK(cudaMemcpy(
      output.data(), thrust::raw_pointer_cast(d_output.data()),
      total_entries * sizeof(Entry), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      live_counts.data(), thrust::raw_pointer_cast(d_live_counts.data()),
      options.vertices * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      overflow.data(), thrust::raw_pointer_cast(d_overflow.data()),
      options.vertices * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  const unsigned int overflow_segments = static_cast<unsigned int>(
      std::count_if(overflow.begin(), overflow.end(),
                    [](unsigned int value) { return value != 0; }));
  const bool correct = validate_output(
      options, reference, output, live_counts, overflow);
  const double update_mups =
      unique_updates / (static_cast<double>(update_ms) * 1000.0);
  const double approximate_bytes_per_update =
      (4.0 * affected_count * options.segment_capacity * sizeof(Entry)) /
      static_cast<double>(unique_updates);

  std::fprintf(stderr, "# gpu=%s initial_live=%u\n",
               properties.name, initial_live);
  std::puts(
      "benchmark,mode,parallel_threshold,serial_segments,parallel_segments,"
      "vertices,segment_capacity,density,input_updates,"
      "unique_updates,affected_segments,distribution,insert_ratio,"
      "duplicate_ratio,preprocess_ms,update_ms,update_mups,"
      "approx_bytes_per_update,overflow_segments,correct");
  std::printf(
      "one_level,%s,%u,%u,%u,%u,%u,%.4f,%zu,%zu,%u,%s,%.4f,%.4f,"
      "%.6f,%.6f,%.6f,%.2f,%u,%s\n",
      options.mode.c_str(), options.parallel_threshold, serial_count,
      parallel_count, options.vertices, options.segment_capacity,
      options.density, options.batch_size, unique_updates, affected_count,
      options.distribution.c_str(), options.insert_ratio,
      options.duplicate_ratio, preprocess_ms, update_ms, update_mups,
      approximate_bytes_per_update, overflow_segments,
      correct ? "true" : "false");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return correct ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run_benchmark(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "update_bench: %s\n", error.what());
    return 1;
  }
}
