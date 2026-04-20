#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Fused bias + GELU kernel
__global__ void bias_gelu_kernel(
    float* data,           // in-place: MatMul output, also output
    const float* bias,     // bias vector [N]
    int M, int N)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;//avoiud overflow

    //bias
    int col = idx % N;
    float x = data[idx] + bias[col];

    // GELU
    const float k = 0.7978845608f;
    float tanh_val = tanhf(k * (x + 0.044715f * x * x * x));
    tanh_val = fmaxf(-1.0f, fminf(1.0f, tanh_val));
    data[idx] = x * 0.5f * (1.0f + tanh_val);
}

//init (connecting to cublas (handle))
static cublasHandle_t g_cublas_handle = nullptr;
void init_cublas() {
    if (!g_cublas_handle) {
        cublasCreate(&g_cublas_handle);
    }
}

void launch_fused_gemm_gelu_fp32(
    const float* A,      // input [M, K]
    const float* B,      // weight [K, N]  
    const float* bias,   // bias [N]
    float* C,            // output [M, N]
    int M, int K, int N,
    cudaStream_t stream)
{
    init_cublas();
    cublasSetStream(g_cublas_handle, stream);//ensure they are in the right stream

    // C = A * B^T using cuBLAS (column-major, so we do B*A^T)
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, K,
        A, K,
        &beta,
        C, N);

    // Fused bias + GELU
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    bias_gelu_kernel<<<blocks, threads, 0, stream>>>(C, bias, M, N);
}