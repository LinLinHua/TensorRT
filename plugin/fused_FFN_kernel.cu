#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

/*
 * Fused GEMM + Bias + GELU kernel for BERT FFN layer.
 * Weight in ONNX is already transposed [in_features, out_features] = [768, 3072]
 * because PyTorch folds the transpose into the ONNX export.
 * So we use CUBLAS_OP_N (no transpose) for both A and B.
 */

// FP32 in-place bias + GELU kernel
__global__ void bias_gelu_kernel_fp32(
    float* data,        // in-place: GEMM output, also final output
    const float* bias,  // bias vector [N]
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int col = idx % N;
    float x = data[idx] + bias[col];

    const float k = 0.7978845608f;
    float tanh_val = tanhf(k * (x + 0.044715f * x * x * x));
    tanh_val = fmaxf(-1.0f, fminf(1.0f, tanh_val));
    data[idx] = x * 0.5f * (1.0f + tanh_val);
}

// FP16 in-place bias + GELU kernel (compute in FP32 for numerical stability)
__global__ void bias_gelu_kernel_fp16(
    __half* data,        // in-place
    const __half* bias,
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int col = idx % N;
    float x = __half2float(data[idx]) + __half2float(bias[col]);

    const float k = 0.7978845608f;
    float tanh_val = tanhf(k * (x + 0.044715f * x * x * x));
    tanh_val = fmaxf(-1.0f, fminf(1.0f, tanh_val));
    float result = x * 0.5f * (1.0f + tanh_val);
    result = fmaxf(-65504.0f, fminf(65504.0f, result));
    data[idx] = __float2half(result);
}

// Global cuBLAS handle — initialized once, reused across all FFN layers
static cublasHandle_t g_cublas_handle = nullptr;

void init_cublas() {
    if (!g_cublas_handle) {
        cublasCreate(&g_cublas_handle);
    }
}

void launch_fused_gemm_gelu_fp32(
    const float* A,     // input [M, K]
    const float* B,     // weight [K, N] (already transposed in ONNX)
    const float* bias,  // bias [N]
    float* C,           // output [M, N]
    int M, int K, int N,
    cudaStream_t stream)
{
    init_cublas();
    cublasSetStream(g_cublas_handle, stream);

    /*
     * C = A * B using cuBLAS column-major convention.
     * Weight B is [K, N] = [768, 3072] (already transposed by PyTorch export).
     * Use CUBLAS_OP_N for both — no additional transpose needed.
     * cuBLAS column-major: compute C^T = B^T * A^T
     * which gives us C = A * B in row-major.
     */
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,   // B [K, N], leading dim = N
        A, K,   // A [M, K], leading dim = K
        &beta,
        C, N);  // C [M, N], leading dim = N

    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    bias_gelu_kernel_fp32<<<blocks, threads, 0, stream>>>(C, bias, M, N);
}

void launch_fused_gemm_gelu_fp16(
    const __half* A,
    const __half* B,
    const __half* bias,
    __half* C,
    int M, int K, int N,
    cudaStream_t stream)
{
    init_cublas();
    cublasSetStream(g_cublas_handle, stream);

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    // Same logic as FP32: B is already transposed, use CUBLAS_OP_N
    cublasHgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,   // B [K, N], leading dim = N
        A, K,   // A [M, K], leading dim = K
        &beta,
        C, N);  // C [M, N], leading dim = N

    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    bias_gelu_kernel_fp16<<<blocks, threads, 0, stream>>>(C, bias, M, N);
}