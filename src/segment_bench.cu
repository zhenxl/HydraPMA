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

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(status_));                     \
    }                                                                            \
  } while (0)

enum class Mode { kScalar, kCpAsync, kTma };

struct Options {
  std::size_t segment_bytes = 4096;
  double density = 0.75;
  std::size_t working_set_mb = 64;
  int warmup = 5;
  int iterations = 20;
  std::string mode = "all";
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

__device__ __forceinline__ void bulk_store_commit_wait() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
  asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
#endif

__host__ __device__ __forceinline__ std::size_t redistributed_position(
    std::size_t live_index, std::size_t capacity, std::size_t live_count) {
  return (live_index * capacity) / live_count;
}

__global__ void initialize_input(Entry* input, Entry* output,
                                 std::size_t total_entries,
                                 std::size_t capacity,
                                 std::size_t live_count) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < total_entries; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t local = i % capacity;
    input[i] = local < live_count ? Entry{i + 1, (i + 1) ^ 0x5a5a5a5aULL}
                                  : Entry{kEmpty, 0};
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
    } else if (!std::strcmp(argv[i], "--help")) {
      std::puts("segment_bench [--segment-bytes N] [--density F] "
                "[--working-set-mb N] [--warmup N] [--iterations N] "
                "[--mode all|scalar|cp_async|tma]");
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
  }
  return "unknown";
}

bool requested(const Options& options, Mode mode) {
  return options.mode == "all" || options.mode == mode_name(mode) ||
         (mode == Mode::kTma && options.mode == "tma");
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

    if (smem_bytes > prop.sharedMemPerBlockOptin) {
      throw std::runtime_error("requested two-buffer shared memory exceeds "
                               "device opt-in limit");
    }

    Entry* input = nullptr;
    Entry* output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, total_entries * sizeof(Entry)));
    CUDA_CHECK(cudaMalloc(&output, total_entries * sizeof(Entry)));
    initialize_input<<<std::min<int>(65535, (total_entries + 255) / 256), 256>>>(
        input, output, total_entries, capacity, live_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::puts("device,cc,mode,segment_bytes,density,segments,live_entries,"
              "latency_ms,effective_gbps,valid");
    const Mode modes[] = {Mode::kScalar, Mode::kCpAsync, Mode::kTma};
    for (Mode mode : modes) {
      if (!requested(options, mode)) {
        continue;
      }
      if (mode == Mode::kCpAsync && prop.major < 8) {
        continue;
      }
      if (mode == Mode::kTma && prop.major < 9) {
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
      } else {
        latency_ms = time_mode<Mode::kTma>(
            input, output, segments, capacity, live_count, smem_bytes,
            options.warmup, options.iterations);
      }
      const bool valid = validate_first_segment(output, capacity, live_count);
      const double bytes = 2.0 * segments * options.segment_bytes;
      const double gbps = bytes / (latency_ms * 1.0e6);
      std::printf("\"%s\",%d.%d,%s,%zu,%.4f,%d,%zu,%.6f,%.3f,%d\n",
                  prop.name, prop.major, prop.minor, mode_name(mode),
                  options.segment_bytes, options.density, segments, live_count,
                  latency_ms, gbps, valid ? 1 : 0);
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
