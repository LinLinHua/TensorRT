#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ inline float gelu_f32(float x) {//compute
    const float k = 0.7978845608f;
    float tanh_val = tanhf(k * (x + 0.044715f * x * x * x));
    tanh_val = fmaxf(-1.0f, fminf(1.0f, tanh_val));//clamp since we use the approximation
    return x * 0.5f * (1.0f + tanh_val);
}

__device__ inline __half gelu_f16(__half x) {//we save the storage and only gelu part is fp32 compute
    float xf = __half2float(x);//up to fp32
    float result = gelu_f32(xf);//calculate
    result = fmaxf(-65504.0f, fminf(65504.0f, result));
    return __float2half(result);
}

// Original GELU kernels (no bias) save to GPU
__global__ void gelu_kernel_fp32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = gelu_f32(input[idx]);//save to GPU
}

__global__ void gelu_kernel_fp16(const __half* input, __half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = gelu_f16(input[idx]);
}


void launch_gelu_fp32(const float* input, float* output, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel_fp32<<<blocks, threads, 0, stream>>>(input, output, n);
}

void launch_gelu_fp16(const __half* input, __half* output, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel_fp16<<<blocks, threads, 0, stream>>>(input, output, n);
}
