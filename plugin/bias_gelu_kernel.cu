#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
Bert model has FFN layer;
FFN layer＝Linear(GEMM+bias)+GELU+Linear

Linear=Linear(x) = x @ weight + bias
                   ↑            ↑
                  GEMM         bias

input>GEMM>+bias>GELU>output

Here: bias+GELU

*/
__device__ inline float gelu_f32(float x) {
    const float k = 0.7978845608f;
    float tanh_val = tanhf(k * (x + 0.044715f * x * x * x));
    tanh_val = fmaxf(-1.0f, fminf(1.0f, tanh_val));
    return x * 0.5f * (1.0f + tanh_val);
}

__device__ inline __half gelu_f16(__half x) {
    float xf = __half2float(x);
    float result = gelu_f32(xf);
    result = fmaxf(-65504.0f, fminf(65504.0f, result));
    return __float2half(result);
}

// Fused bias + GELU kernels
__global__ void gelu_bias_kernel_fp32(
    const float* input, const float* bias, float* output, int n, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int col = idx % hidden_size;//bias
        output[idx] = gelu_f32(input[idx] + bias[col]);
    }
}

__global__ void gelu_bias_kernel_fp16(
    const __half* input, const __half* bias, __half* output, int n, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int col = idx % hidden_size;//linear bias
        float xf = __half2float(input[idx]) + __half2float(bias[col]);
        float result = gelu_f32(xf);
        result = fmaxf(-65504.0f, fminf(65504.0f, result));
        output[idx] = __float2half(result);
    }
}

void launch_gelu_bias_fp32(
    const float* input, const float* bias, float* output,
    int n, int hidden_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_bias_kernel_fp32<<<blocks, threads, 0, stream>>>(input, bias, output, n, hidden_size);
    //hidden_size for bias layer of FFN
}

void launch_gelu_bias_fp16(
    const __half* input, const __half* bias, __half* output,
    int n, int hidden_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_bias_kernel_fp16<<<blocks, threads, 0, stream>>>(input, bias, output, n, hidden_size);
}