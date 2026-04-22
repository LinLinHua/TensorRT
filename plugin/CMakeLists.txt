cmake_minimum_required(VERSION 3.18)
project(gelu_plugin CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include
    REQUIRED)

find_library(TRT_LIB nvinfer
    PATHS /usr/lib/x86_64-linux-gnu
    REQUIRED)

find_library(CUBLAS_LIB cublas
    PATHS /usr/local/cuda/lib64
    REQUIRED)

include_directories(
    ${TRT_INCLUDE_DIR}
    /usr/local/cuda/include
    /workspace/plugin
)

add_library(gelu_plugin SHARED
    gelu_plugin.cpp
    gelu_kernel.cu
    bias_gelu_kernel.cu
    bias_gelu_plugin.cpp
    fused_FFN_kernel.cu
    fused_FFN_plugin.cpp
)

target_link_libraries(gelu_plugin ${TRT_LIB} cudart ${CUBLAS_LIB})