#include "fused_gemm_gelu_plugin.h"
#include <cstdio>

int32_t FusedGemmGeluPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    // Debug: print input shapes
    fprintf(stderr, "FusedPlugin enqueue called\n");
    fprintf(stderr, "Input0 nbDims=%d dims=", inputDesc[0].dims.nbDims);
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        fprintf(stderr, "%d ", inputDesc[0].dims.d[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "Input1 nbDims=%d dims=", inputDesc[1].dims.nbDims);
    for (int i = 0; i < inputDesc[1].dims.nbDims; i++)
        fprintf(stderr, "%d ", inputDesc[1].dims.d[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "Input2 nbDims=%d dims=", inputDesc[2].dims.nbDims);
    for (int i = 0; i < inputDesc[2].dims.nbDims; i++)
        fprintf(stderr, "%d ", inputDesc[2].dims.d[i]);
    fprintf(stderr, "\n");

    // Compute M, K, N from actual input shapes
    // Input0: [batch, seq, hidden] or [batch*seq, hidden]
    // Input1: [out_features, in_features] (weight is transposed in Linear)
    // Input2: [out_features] (bias)

    // Flatten input0 to 2D: M = product of all dims except last, K = last dim
    int M = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; i++)
        M *= inputDesc[0].dims.d[i];
    int K = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]; // in_features
    // Weight shape: [out_features, in_features]
    int N = inputDesc[1].dims.d[0];

    fprintf(stderr, "M=%d K=%d N=%d\n", M, K, N);

    launch_fused_gemm_gelu_fp32(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<float*>(outputs[0]),
        M, K, N, stream);

    return 0;
}

__attribute__((constructor)) static void registerFusedGemmGeluPlugin() {
    static FusedGemmGeluPluginCreator creator;
    fprintf(stderr, "[FusedGemmGeluPlugin] Registering FusedGemmGeluPlugin...\n");
    getPluginRegistry()->registerCreator(creator, "");
    fprintf(stderr, "[FusedGemmGeluPlugin] Done.\n");
}