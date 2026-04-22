#include "fused_FFN_plugin.h"
#include <cstdio>

int32_t FusedGemmGeluPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    int M = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; i++)
        M *= inputDesc[0].dims.d[i];
    int K = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    int N = inputDesc[1].dims.d[1];

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        launch_fused_gemm_gelu_fp32(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]),
            static_cast<float*>(outputs[0]),
            M, K, N, stream);
    } else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
        launch_fused_gemm_gelu_fp16(
            static_cast<const __half*>(inputs[0]),
            static_cast<const __half*>(inputs[1]),
            static_cast<const __half*>(inputs[2]),
            static_cast<__half*>(outputs[0]),
            M, K, N, stream);
    }

    return 0;
}

REGISTER_TENSORRT_PLUGIN(FusedGemmGeluPluginCreator);