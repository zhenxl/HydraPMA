cmake_minimum_required(VERSION 3.22)

project(hydrapma LANGUAGES CXX CUDA)

# sm_90a is required for Hopper architecture-specific TMA instructions.  CMake's
# native architecture list has historically lagged the "a" targets, so pass the
# nvcc target explicitly.
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(HYDRAPMA_CUDA_ARCH "sm_90a" CACHE STRING "nvcc GPU architecture")

add_executable(segment_bench src/segment_bench.cu)
target_compile_features(segment_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(segment_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${HYDRAPMA_CUDA_ARCH}>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(update_bench src/update_bench.cu)
target_compile_features(update_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(update_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${HYDRAPMA_CUDA_ARCH}>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(lockfree_delta_bench src/lockfree_delta_bench.cu)
target_compile_features(lockfree_delta_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(lockfree_delta_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${HYDRAPMA_CUDA_ARCH}>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
