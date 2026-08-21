#include <cuda_runtime.h>

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
#include <unordered_set>
#include <vector>

namespace {

struct DeltaRecord {
  unsigned int src;
  unsigned int dst;
  unsigned int sequence;
  unsigned int op;
  std::uint64_t value;
};

struct Counters {
  unsigned long long reservation_atomics;
  unsigned long long rollover_attempts;
  unsigned long long cas_retries;
  unsigned long long wasted_blocks;
  unsigned long long pool_exhausted;
};

struct Options {
  unsigned int vertices = 1024;
  unsigned int block_capacity = 32;
  std::size_t batch_size = 65536;
  std::string input_order = "grouped";
  std::string distribution = "uniform";
  std::string mode = "all";
  double insert_ratio = 0.5;
  double zipf_skew = 1.1;
  unsigned int pool_factor = 4;
  std::uint64_t seed = 19;
  int warmup = 2;
  int iterations = 5;
};

struct Result {
  std::string mode;
  float append_ms = 0.0f;
  unsigned int blocks_used = 0;
  unsigned int ready_records = 0;
  Counters counters{};
  bool correct = false;
};

constexpr int kThreads = 256;
constexpr unsigned int kInsert = 1;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(call) check_cuda((call), #call)

__host__ __device__ __forceinline__ unsigned long long make_handle(
    unsigned int generation, unsigned int block_index) {
  return (static_cast<unsigned long long>(generation) << 32) |
         static_cast<unsigned long long>(block_index + 1);
}

__host__ __device__ __forceinline__ unsigned int handle_index(
    unsigned long long handle) {
  return static_cast<unsigned int>(handle) - 1;
}

__host__ __device__ __forceinline__ unsigned int handle_generation(
    unsigned long long handle) {
  return static_cast<unsigned int>(handle >> 32);
}

__global__ void initialize_active_blocks_kernel(
    unsigned int vertices, unsigned long long* active_handles,
    unsigned int* block_owner, unsigned int* block_generation,
    unsigned int* next_block) {
  for (unsigned int src = blockIdx.x * blockDim.x + threadIdx.x;
       src < vertices; src += gridDim.x * blockDim.x) {
    active_handles[src] = make_handle(1, src);
    block_owner[src] = src;
    block_generation[src] = 1;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *next_block = vertices;
  }
}

template <bool warp_grouped>
__global__ void append_delta_kernel(
    const DeltaRecord* updates, std::size_t update_count,
    unsigned long long* active_handles, unsigned int* block_tails,
    unsigned int* block_sealed, unsigned int* block_owner,
    unsigned int* block_generation, unsigned long long* previous_handles,
    DeltaRecord* records, unsigned int* ready, unsigned int block_capacity,
    unsigned int max_blocks, unsigned int* next_block, Counters* counters) {
  const std::size_t update_index =
      std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (update_index >= update_count) {
    return;
  }

  const DeltaRecord update = updates[update_index];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int active_mask = __activemask();
  const unsigned int group_mask =
      warp_grouped ? __match_any_sync(active_mask, update.src)
                   : (1U << lane);
  const unsigned int leader = __ffs(group_mask) - 1;
  const unsigned int lower_lanes =
      lane == 0 ? 0U : ((1U << lane) - 1U);
  const unsigned int group_rank = __popc(group_mask & lower_lanes);
  const unsigned int group_size = __popc(group_mask);

  while (true) {
    const unsigned long long old_handle =
        atomicCAS(active_handles + update.src, 0ULL, 0ULL);
    const unsigned int old_index = handle_index(old_handle);
    unsigned int group_offset = 0;
    if (lane == leader) {
      group_offset =
          atomicAdd(block_tails + old_index, group_size);
      atomicAdd(&counters->reservation_atomics, 1ULL);
    }
    group_offset =
        __shfl_sync(group_mask, group_offset, static_cast<int>(leader));

    if (group_offset + group_size <= block_capacity) {
      const unsigned int slot = group_offset + group_rank;
      const std::size_t record_index =
          std::size_t{old_index} * block_capacity + slot;
      records[record_index] = update;
      __threadfence();
      atomicExch(ready + record_index, 1U);
      return;
    }

    unsigned int abort = 0;
    if (lane == leader) {
      const unsigned long long current =
          atomicCAS(active_handles + update.src, 0ULL, 0ULL);
      if (current == old_handle) {
        atomicAdd(&counters->rollover_attempts, 1ULL);
        const unsigned int candidate = atomicAdd(next_block, 1U);
        if (candidate >= max_blocks) {
          atomicAdd(&counters->pool_exhausted, 1ULL);
          abort = 1;
        } else {
          const unsigned int generation =
              handle_generation(old_handle) + 1U;
          block_tails[candidate] = 0;
          block_sealed[candidate] = 0;
          block_owner[candidate] = update.src;
          block_generation[candidate] = generation;
          previous_handles[candidate] = old_handle;
          __threadfence();
          const unsigned long long new_handle =
              make_handle(generation, candidate);
          const unsigned long long observed = atomicCAS(
              active_handles + update.src, old_handle, new_handle);
          if (observed == old_handle) {
            atomicExch(block_sealed + old_index, 1U);
          } else {
            atomicAdd(&counters->cas_retries, 1ULL);
            atomicAdd(&counters->wasted_blocks, 1ULL);
          }
        }
      }
    }
    abort = __shfl_sync(group_mask, abort, static_cast<int>(leader));
    if (abort != 0) {
      return;
    }
  }
}

std::vector<DeltaRecord> make_updates(const Options& options) {
  if (options.batch_size >
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error("batch exceeds 32-bit sequence range");
  }
  std::mt19937_64 rng(options.seed);
  std::uniform_int_distribution<unsigned int> uniform_src(
      0, options.vertices - 1);
  std::uniform_int_distribution<unsigned int> dst_distribution(
      0, std::numeric_limits<unsigned int>::max() - 1);
  std::bernoulli_distribution insert_distribution(options.insert_ratio);
  std::discrete_distribution<unsigned int> zipf_src;
  if (options.distribution == "zipf") {
    std::vector<double> weights(options.vertices);
    for (unsigned int i = 0; i < options.vertices; ++i) {
      weights[i] =
          1.0 / std::pow(static_cast<double>(i + 1), options.zipf_skew);
    }
    zipf_src = std::discrete_distribution<unsigned int>(
        weights.begin(), weights.end());
  }

  std::vector<DeltaRecord> updates;
  updates.reserve(options.batch_size);
  for (std::size_t i = 0; i < options.batch_size; ++i) {
    const unsigned int src =
        options.distribution == "zipf" ? zipf_src(rng) : uniform_src(rng);
    const unsigned int dst = dst_distribution(rng);
    const unsigned int sequence = static_cast<unsigned int>(i);
    const unsigned int op =
        insert_distribution(rng) ? kInsert : 0U;
    const std::uint64_t value =
        (std::uint64_t{sequence} << 32) ^
        (std::uint64_t{src} * 0x9e3779b97f4a7c15ULL) ^ dst;
    updates.push_back(DeltaRecord{src, dst, sequence, op, value});
  }
  if (options.input_order == "grouped") {
    std::stable_sort(
        updates.begin(), updates.end(),
        [](const DeltaRecord& a, const DeltaRecord& b) {
          return a.src < b.src;
        });
  }
  return updates;
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
    } else if (!std::strcmp(argv[i], "--block-capacity")) {
      options.block_capacity = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--batch-size")) {
      options.batch_size = std::stoull(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--input-order")) {
      options.input_order = value(argv[i]);
    } else if (!std::strcmp(argv[i], "--distribution")) {
      options.distribution = value(argv[i]);
    } else if (!std::strcmp(argv[i], "--mode")) {
      options.mode = value(argv[i]);
    } else if (!std::strcmp(argv[i], "--insert-ratio")) {
      options.insert_ratio = std::stod(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--zipf-skew")) {
      options.zipf_skew = std::stod(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--pool-factor")) {
      options.pool_factor = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--seed")) {
      options.seed = std::stoull(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      options.warmup = std::stoi(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--iterations")) {
      options.iterations = std::stoi(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--help")) {
      std::puts(
          "lockfree_delta_bench [--vertices N] [--block-capacity N] "
          "[--batch-size N] [--input-order grouped|random] "
          "[--distribution uniform|zipf] [--mode all|atomic|warp] "
          "[--insert-ratio F] [--zipf-skew F] [--pool-factor N] "
          "[--seed N] [--warmup N] [--iterations N]");
      std::exit(0);
    } else {
      throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
    }
  }

  if (options.vertices == 0 || options.block_capacity == 0 ||
      options.batch_size == 0 || options.pool_factor == 0) {
    throw std::runtime_error("sizes and pool factor must be nonzero");
  }
  if (options.input_order != "grouped" &&
      options.input_order != "random") {
    throw std::runtime_error("input order must be grouped or random");
  }
  if (options.distribution != "uniform" &&
      options.distribution != "zipf") {
    throw std::runtime_error("distribution must be uniform or zipf");
  }
  if (options.mode != "all" && options.mode != "atomic" &&
      options.mode != "warp") {
    throw std::runtime_error("mode must be all, atomic, or warp");
  }
  if ((options.mode == "all" || options.mode == "warp") &&
      options.block_capacity < 32) {
    throw std::runtime_error(
        "warp mode requires block capacity of at least 32");
  }
  if (!(options.insert_ratio >= 0.0 && options.insert_ratio <= 1.0) ||
      options.zipf_skew <= 0.0 || options.warmup < 0 ||
      options.iterations <= 0) {
    throw std::runtime_error("invalid ratio, skew, or iteration count");
  }
  return options;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

void reset_structure(
    const Options& options, unsigned int max_blocks,
    DeviceBuffer<unsigned long long>& active_handles,
    DeviceBuffer<unsigned int>& block_tails,
    DeviceBuffer<unsigned int>& block_sealed,
    DeviceBuffer<unsigned int>& block_owner,
    DeviceBuffer<unsigned int>& block_generation,
    DeviceBuffer<unsigned long long>& previous_handles,
    DeviceBuffer<unsigned int>& ready, DeviceBuffer<unsigned int>& next_block,
    DeviceBuffer<Counters>& counters) {
  CUDA_CHECK(cudaMemset(block_tails.data(), 0,
                        max_blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(block_sealed.data(), 0,
                        max_blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(block_owner.data(), 0xff,
                        max_blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(block_generation.data(), 0,
                        max_blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(previous_handles.data(), 0,
                        max_blocks * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(ready.data(), 0,
                        ready.size() * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(counters.data(), 0, sizeof(Counters)));

  const int blocks = static_cast<int>(
      (std::size_t{options.vertices} + kThreads - 1) / kThreads);
  initialize_active_blocks_kernel<<<blocks, kThreads>>>(
      options.vertices, active_handles.data(), block_owner.data(),
      block_generation.data(), next_block.data());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

bool same_record(const DeltaRecord& a, const DeltaRecord& b) {
  return a.src == b.src && a.dst == b.dst &&
         a.sequence == b.sequence && a.op == b.op && a.value == b.value;
}

bool validate_structure(
    const Options& options, const std::vector<DeltaRecord>& updates,
    unsigned int max_blocks,
    const DeviceBuffer<unsigned long long>& active_handles,
    const DeviceBuffer<unsigned int>& block_tails,
    const DeviceBuffer<unsigned int>& block_sealed,
    const DeviceBuffer<unsigned int>& block_owner,
    const DeviceBuffer<unsigned int>& block_generation,
    const DeviceBuffer<unsigned long long>& previous_handles,
    const DeviceBuffer<DeltaRecord>& records,
    const DeviceBuffer<unsigned int>& ready, unsigned int blocks_used,
    const Counters& counters, unsigned int* ready_records) {
  std::vector<unsigned long long> host_active(options.vertices);
  std::vector<unsigned int> host_tails(max_blocks);
  std::vector<unsigned int> host_sealed(max_blocks);
  std::vector<unsigned int> host_owner(max_blocks);
  std::vector<unsigned int> host_generation(max_blocks);
  std::vector<unsigned long long> host_previous(max_blocks);
  std::vector<DeltaRecord> host_records(records.size());
  std::vector<unsigned int> host_ready(ready.size());
  CUDA_CHECK(cudaMemcpy(host_active.data(), active_handles.data(),
                        host_active.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_tails.data(), block_tails.data(),
                        host_tails.size() * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_sealed.data(), block_sealed.data(),
                        host_sealed.size() * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_owner.data(), block_owner.data(),
                        host_owner.size() * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_generation.data(), block_generation.data(),
                        host_generation.size() * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_previous.data(), previous_handles.data(),
                        host_previous.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_records.data(), records.data(),
                        host_records.size() * sizeof(DeltaRecord),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_ready.data(), ready.data(),
                        host_ready.size() * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  unsigned int global_ready = 0;
  for (unsigned int flag : host_ready) {
    global_ready += flag != 0 ? 1U : 0U;
  }
  *ready_records = global_ready;
  if (counters.pool_exhausted != 0 || blocks_used > max_blocks ||
      global_ready != updates.size()) {
    return false;
  }

  std::vector<DeltaRecord> by_sequence(updates.size());
  std::vector<std::vector<unsigned int>> expected(options.vertices);
  for (const DeltaRecord& update : updates) {
    by_sequence[update.sequence] = update;
    expected[update.src].push_back(update.sequence);
  }
  for (auto& sequences : expected) {
    std::sort(sequences.begin(), sequences.end());
  }

  unsigned int chained_ready = 0;
  for (unsigned int src = 0; src < options.vertices; ++src) {
    std::vector<unsigned int> actual;
    std::unordered_set<unsigned int> visited;
    unsigned long long handle = host_active[src];
    bool active = true;
    while (handle != 0) {
      const unsigned int block = handle_index(handle);
      if (block >= blocks_used || block >= max_blocks ||
          !visited.insert(block).second ||
          host_owner[block] != src ||
          host_generation[block] != handle_generation(handle) ||
          (active ? host_sealed[block] != 0
                  : host_sealed[block] == 0)) {
        return false;
      }
      for (unsigned int slot = 0; slot < options.block_capacity; ++slot) {
        const std::size_t index =
            std::size_t{block} * options.block_capacity + slot;
        if (host_ready[index] == 0) {
          continue;
        }
        if (slot >= std::min(host_tails[block],
                             options.block_capacity)) {
          return false;
        }
        const DeltaRecord& record = host_records[index];
        if (record.sequence >= by_sequence.size() ||
            record.src != src ||
            !same_record(record, by_sequence[record.sequence])) {
          return false;
        }
        actual.push_back(record.sequence);
        ++chained_ready;
      }
      handle = host_previous[block];
      active = false;
    }
    std::sort(actual.begin(), actual.end());
    if (actual != expected[src]) {
      return false;
    }
  }
  return chained_ready == global_ready;
}

template <bool warp_grouped>
Result run_mode(
    const Options& options, const std::vector<DeltaRecord>& updates,
    unsigned int max_blocks, DeviceBuffer<DeltaRecord>& device_updates,
    DeviceBuffer<unsigned long long>& active_handles,
    DeviceBuffer<unsigned int>& block_tails,
    DeviceBuffer<unsigned int>& block_sealed,
    DeviceBuffer<unsigned int>& block_owner,
    DeviceBuffer<unsigned int>& block_generation,
    DeviceBuffer<unsigned long long>& previous_handles,
    DeviceBuffer<DeltaRecord>& records, DeviceBuffer<unsigned int>& ready,
    DeviceBuffer<unsigned int>& next_block,
    DeviceBuffer<Counters>& counters) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const int blocks = static_cast<int>(
      (updates.size() + kThreads - 1) / kThreads);
  float measured_ms = 0.0f;

  for (int iteration = -options.warmup;
       iteration < options.iterations; ++iteration) {
    reset_structure(options, max_blocks, active_handles, block_tails,
                    block_sealed, block_owner, block_generation,
                    previous_handles, ready, next_block, counters);
    CUDA_CHECK(cudaEventRecord(start));
    append_delta_kernel<warp_grouped><<<blocks, kThreads>>>(
        device_updates.data(), updates.size(), active_handles.data(),
        block_tails.data(), block_sealed.data(), block_owner.data(),
        block_generation.data(), previous_handles.data(), records.data(),
        ready.data(), options.block_capacity, max_blocks,
        next_block.data(), counters.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (iteration >= 0) {
      float elapsed = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
      measured_ms += elapsed;
    }
  }

  Result result;
  result.mode = warp_grouped ? "warp" : "atomic";
  result.append_ms = measured_ms / options.iterations;
  CUDA_CHECK(cudaMemcpy(&result.blocks_used, next_block.data(),
                        sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.counters, counters.data(),
                        sizeof(Counters), cudaMemcpyDeviceToHost));
  result.correct = validate_structure(
      options, updates, max_blocks, active_handles, block_tails,
      block_sealed, block_owner, block_generation, previous_handles,
      records, ready, result.blocks_used, result.counters,
      &result.ready_records);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return result;
}

int run_benchmark(const Options& options) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const std::vector<DeltaRecord> updates = make_updates(options);

  const std::size_t minimum_rollovers =
      (options.batch_size + options.block_capacity - 1) /
          options.block_capacity +
      options.vertices;
  if (minimum_rollovers >
      (std::numeric_limits<std::size_t>::max() - options.vertices) /
          options.pool_factor) {
    throw std::runtime_error("delta pool size overflows size_t");
  }
  const std::size_t max_blocks_size =
      options.vertices + options.pool_factor * minimum_rollovers;
  if (max_blocks_size >
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error("delta block count exceeds 32-bit handle range");
  }
  const unsigned int max_blocks =
      static_cast<unsigned int>(max_blocks_size);
  if (max_blocks_size >
      std::numeric_limits<std::size_t>::max() / options.block_capacity) {
    throw std::runtime_error("delta record allocation overflows size_t");
  }
  const std::size_t record_slots =
      max_blocks_size * options.block_capacity;

  DeviceBuffer<DeltaRecord> device_updates(updates.size());
  DeviceBuffer<unsigned long long> active_handles(options.vertices);
  DeviceBuffer<unsigned int> block_tails(max_blocks);
  DeviceBuffer<unsigned int> block_sealed(max_blocks);
  DeviceBuffer<unsigned int> block_owner(max_blocks);
  DeviceBuffer<unsigned int> block_generation(max_blocks);
  DeviceBuffer<unsigned long long> previous_handles(max_blocks);
  DeviceBuffer<DeltaRecord> records(record_slots);
  DeviceBuffer<unsigned int> ready(record_slots);
  DeviceBuffer<unsigned int> next_block(1);
  DeviceBuffer<Counters> counters(1);
  CUDA_CHECK(cudaMemcpy(device_updates.data(), updates.data(),
                        updates.size() * sizeof(DeltaRecord),
                        cudaMemcpyHostToDevice));

  std::vector<Result> results;
  if (options.mode == "all" || options.mode == "atomic") {
    results.push_back(run_mode<false>(
        options, updates, max_blocks, device_updates, active_handles,
        block_tails, block_sealed, block_owner, block_generation,
        previous_handles, records, ready, next_block, counters));
  }
  if (options.mode == "all" || options.mode == "warp") {
    results.push_back(run_mode<true>(
        options, updates, max_blocks, device_updates, active_handles,
        block_tails, block_sealed, block_owner, block_generation,
        previous_handles, records, ready, next_block, counters));
  }

  std::fprintf(stderr, "# gpu=%s max_blocks=%u record_slots=%zu\n",
               properties.name, max_blocks, record_slots);
  std::puts(
      "benchmark,mode,input_order,distribution,vertices,block_capacity,"
      "batch_size,blocks_used,reservation_atomics,rollover_attempts,"
      "cas_retries,wasted_blocks,pool_exhausted,append_ms,append_mups,"
      "reservation_atomics_per_update,ready_records,correct");
  bool all_correct = true;
  for (const Result& result : results) {
    const double append_mups =
        options.batch_size /
        (static_cast<double>(result.append_ms) * 1000.0);
    const double atomics_per_update =
        static_cast<double>(result.counters.reservation_atomics) /
        options.batch_size;
    std::printf(
        "lockfree_delta,%s,%s,%s,%u,%u,%zu,%u,%llu,%llu,%llu,%llu,"
        "%llu,%.6f,%.6f,%.6f,%u,%s\n",
        result.mode.c_str(), options.input_order.c_str(),
        options.distribution.c_str(), options.vertices,
        options.block_capacity, options.batch_size, result.blocks_used,
        result.counters.reservation_atomics,
        result.counters.rollover_attempts, result.counters.cas_retries,
        result.counters.wasted_blocks, result.counters.pool_exhausted,
        result.append_ms, append_mups, atomics_per_update,
        result.ready_records, result.correct ? "true" : "false");
    all_correct = all_correct && result.correct;
  }
  return all_correct ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run_benchmark(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "lockfree_delta_bench: %s\n", error.what());
    return 1;
  }
}
