#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SegmentDesc {
  std::uint64_t base_offset;
  unsigned int count;
  unsigned int generation;
};

struct QueryCounters {
  unsigned long long accepted;
  unsigned long long retries;
  unsigned long long mismatches;
};

struct Options {
  unsigned int segments = 256;
  unsigned int capacity = 256;
  unsigned int epochs = 32;
  unsigned int query_blocks = 0;
  int repetitions = 5;
};

constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(call) check_cuda((call), #call)

__host__ __device__ __forceinline__ unsigned long long make_handle(
    unsigned int generation, unsigned int descriptor_index) {
  return (static_cast<unsigned long long>(generation) << 32) |
         static_cast<unsigned long long>(descriptor_index + 1);
}

__host__ __device__ __forceinline__ unsigned int handle_index(
    unsigned long long handle) {
  return static_cast<unsigned int>(handle) - 1;
}

__host__ __device__ __forceinline__ std::uint64_t expected_value(
    unsigned int segment, unsigned int generation, unsigned int index) {
  return (std::uint64_t{generation} << 48) ^
         (std::uint64_t{segment} << 24) ^ index ^
         0x9e3779b97f4a7c15ULL;
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

__global__ void build_and_publish_kernel(
    SegmentDesc* descriptors, std::uint64_t* bases,
    unsigned long long* active_handles, unsigned int segments,
    unsigned int capacity, unsigned int epoch, unsigned int* query_started,
    unsigned long long* cas_failures) {
  if (epoch != 0) {
    while (atomicAdd(query_started, 0U) == 0U) {
    }
  }
  const unsigned int segment = blockIdx.x;
  const unsigned int descriptor_index = epoch * segments + segment;
  const std::uint64_t base_offset =
      std::uint64_t{descriptor_index} * capacity;
  const unsigned int generation = epoch + 1;

  for (unsigned int index = threadIdx.x; index < capacity;
       index += blockDim.x) {
    bases[base_offset + index] =
        expected_value(segment, generation, index);
  }
  if (threadIdx.x == 0) {
    descriptors[descriptor_index] =
        SegmentDesc{base_offset, capacity, generation};
  }
  __threadfence();
  __syncthreads();

  if (threadIdx.x == 0) {
    const unsigned long long expected =
        epoch == 0
            ? 0ULL
            : make_handle(epoch, (epoch - 1) * segments + segment);
    const unsigned long long desired =
        make_handle(generation, descriptor_index);
    const unsigned long long observed = atomicCAS(
        active_handles + segment, expected, desired);
    if (observed != expected) {
      atomicAdd(cas_failures, 1ULL);
    }
  }
}

__global__ void query_snapshots_kernel(
    const SegmentDesc* descriptors, const std::uint64_t* bases,
    unsigned long long* active_handles, unsigned int segments,
    unsigned int* query_started, unsigned int* publication_done,
    QueryCounters* counters) {
  __shared__ unsigned long long first_handle;
  __shared__ SegmentDesc descriptor;
  __shared__ unsigned int mismatch;
  __shared__ unsigned int stop;
  __shared__ unsigned int done_seen;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicExch(query_started, 1U);
  }
  if (threadIdx.x == 0) {
    done_seen = 0;
  }
  __syncthreads();

  while (true) {
    for (unsigned int segment = blockIdx.x; segment < segments;
         segment += gridDim.x) {
      if (threadIdx.x == 0) {
        first_handle =
            atomicCAS(active_handles + segment, 0ULL, 0ULL);
        descriptor = descriptors[handle_index(first_handle)];
        mismatch = 0;
      }
      __syncthreads();

      for (unsigned int index = threadIdx.x; index < descriptor.count;
           index += blockDim.x) {
        const std::uint64_t actual =
            bases[descriptor.base_offset + index];
        if (actual != expected_value(segment, descriptor.generation, index)) {
          atomicExch(&mismatch, 1U);
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        const unsigned long long second_handle =
            atomicCAS(active_handles + segment, 0ULL, 0ULL);
        if (first_handle == second_handle) {
          atomicAdd(&counters->accepted, 1ULL);
          if (mismatch != 0) {
            atomicAdd(&counters->mismatches, 1ULL);
          }
        } else {
          atomicAdd(&counters->retries, 1ULL);
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (done_seen != 0) {
        stop = 1;
      } else {
        done_seen = atomicAdd(publication_done, 0U);
        stop = 0;
      }
    }
    __syncthreads();
    if (stop != 0) {
      break;
    }
  }
}

__global__ void mark_done_kernel(unsigned int* publication_done) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicExch(publication_done, 1U);
  }
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
    if (!std::strcmp(argv[i], "--segments")) {
      options.segments = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--capacity")) {
      options.capacity = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--epochs")) {
      options.epochs = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--query-blocks")) {
      options.query_blocks = std::stoul(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--repetitions")) {
      options.repetitions = std::stoi(value(argv[i]));
    } else if (!std::strcmp(argv[i], "--help")) {
      std::puts(
          "publication_bench [--segments N] [--capacity N] [--epochs N] "
          "[--query-blocks N] [--repetitions N]");
      std::exit(0);
    } else {
      throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
    }
  }
  if (options.segments == 0 || options.capacity == 0 ||
      options.epochs == 0 || options.repetitions <= 0) {
    throw std::runtime_error("sizes, epochs, and repetitions must be nonzero");
  }
  return options;
}

int run_benchmark(const Options& options) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  const std::size_t generations =
      static_cast<std::size_t>(options.epochs) + 1;
  if (generations >
      std::numeric_limits<std::size_t>::max() / options.segments) {
    throw std::runtime_error("descriptor count overflows size_t");
  }
  const std::size_t descriptor_count = generations * options.segments;
  if (descriptor_count >
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error("descriptor handle exceeds 32-bit index range");
  }
  if (descriptor_count >
      std::numeric_limits<std::size_t>::max() / options.capacity) {
    throw std::runtime_error("replacement base allocation overflows size_t");
  }
  const std::size_t base_entries = descriptor_count * options.capacity;

  DeviceBuffer<SegmentDesc> descriptors(descriptor_count);
  DeviceBuffer<std::uint64_t> bases(base_entries);
  DeviceBuffer<unsigned long long> active_handles(options.segments);
  DeviceBuffer<unsigned int> query_started(1);
  DeviceBuffer<unsigned int> publication_done(1);
  DeviceBuffer<QueryCounters> query_counters(1);
  DeviceBuffer<unsigned long long> cas_failures(1);

  const unsigned int safe_query_blocks = std::max(
      1, properties.multiProcessorCount / 2);
  const unsigned int query_blocks =
      std::min(options.segments,
               options.query_blocks == 0
                   ? safe_query_blocks
                   : std::min(options.query_blocks, safe_query_blocks));
  cudaStream_t query_stream{};
  cudaStream_t publisher_stream{};
  CUDA_CHECK(cudaStreamCreate(&query_stream));
  CUDA_CHECK(cudaStreamCreate(&publisher_stream));
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  float total_publish_ms = 0.0f;
  QueryCounters total_queries{};
  unsigned long long total_cas_failures = 0;
  bool correct = true;
  for (int repetition = 0; repetition < options.repetitions; ++repetition) {
    CUDA_CHECK(cudaMemset(active_handles.data(), 0,
                          active_handles.size() *
                              sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(query_started.data(), 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(publication_done.data(), 0,
                          sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(query_counters.data(), 0,
                          sizeof(QueryCounters)));
    CUDA_CHECK(cudaMemset(cas_failures.data(), 0,
                          sizeof(unsigned long long)));

    build_and_publish_kernel<<<options.segments, kThreads>>>(
        descriptors.data(), bases.data(), active_handles.data(),
        options.segments, options.capacity, 0, query_started.data(),
        cas_failures.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    query_snapshots_kernel<<<query_blocks, kThreads, 0, query_stream>>>(
        descriptors.data(), bases.data(), active_handles.data(),
        options.segments, query_started.data(), publication_done.data(),
        query_counters.data());
    CUDA_CHECK(cudaEventRecord(start, publisher_stream));
    for (unsigned int epoch = 1; epoch <= options.epochs; ++epoch) {
      build_and_publish_kernel<<<options.segments, kThreads, 0,
                                 publisher_stream>>>(
          descriptors.data(), bases.data(), active_handles.data(),
          options.segments, options.capacity, epoch,
          query_started.data(), cas_failures.data());
    }
    mark_done_kernel<<<1, 1, 0, publisher_stream>>>(
        publication_done.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, publisher_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaStreamSynchronize(query_stream));

    float publish_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&publish_ms, start, stop));
    total_publish_ms += publish_ms;
    QueryCounters host_queries{};
    unsigned long long host_cas_failures = 0;
    CUDA_CHECK(cudaMemcpy(&host_queries, query_counters.data(),
                          sizeof(QueryCounters), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_cas_failures, cas_failures.data(),
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    total_queries.accepted += host_queries.accepted;
    total_queries.retries += host_queries.retries;
    total_queries.mismatches += host_queries.mismatches;
    total_cas_failures += host_cas_failures;

    std::vector<unsigned long long> host_active(options.segments);
    CUDA_CHECK(cudaMemcpy(host_active.data(), active_handles.data(),
                          host_active.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    for (unsigned int segment = 0; segment < options.segments; ++segment) {
      const unsigned int final_index =
          options.epochs * options.segments + segment;
      if (host_active[segment] !=
          make_handle(options.epochs + 1, final_index)) {
        correct = false;
      }
    }
    correct = correct && host_queries.accepted >= options.segments &&
              host_queries.mismatches == 0 &&
              host_cas_failures == 0;
  }

  const double average_ms = total_publish_ms / options.repetitions;
  const double publications =
      static_cast<double>(options.epochs) * options.segments;
  const double publications_mps =
      publications / (average_ms * 1000.0);
  std::fprintf(stderr, "# gpu=%s descriptors=%zu base_entries=%zu\n",
               properties.name, descriptor_count, base_entries);
  std::puts(
      "benchmark,segments,capacity,epochs,query_blocks,repetitions,"
      "publish_ms,publications_mps,accepted_snapshots,query_retries,"
      "snapshot_mismatches,cas_failures,correct");
  std::printf(
      "cow_publication,%u,%u,%u,%u,%d,%.6f,%.6f,%llu,%llu,%llu,%llu,%s\n",
      options.segments, options.capacity, options.epochs, query_blocks,
      options.repetitions, average_ms, publications_mps,
      total_queries.accepted, total_queries.retries,
      total_queries.mismatches, total_cas_failures,
      correct ? "true" : "false");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(query_stream));
  CUDA_CHECK(cudaStreamDestroy(publisher_stream));
  return correct ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run_benchmark(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "publication_bench: %s\n", error.what());
    return 1;
  }
}
