cmake_minimum_required(VERSION 3.10)
project(gelu C CXX CUDA)

find_package(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_executable(gelu
    src/host_main.cpp
    src/kernels/kernel_baseline.cu
)

target_include_directories(gelu PRIVATE ${CUDA_INCLUDE_DIRS})

target_compile_options(gelu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++14;-O3;-lineinfo;--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-arch=sm_80>
)

set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "Target CUDA architectures")

target_link_libraries(gelu PRIVATE
    ${CUDA_LIBRARIES}
)

set_target_properties(gelu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)