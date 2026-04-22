#include "gelu_plugin.h"
#include "bias_gelu_plugin.h"
#include "fused_FFN_plugin.h"
#include <cstdio>
#include <NvInferRuntimePlugin.h>

// ─── GeluPluginV3 enqueue ────────────────────────────────────────────────────

int32_t GeluPluginV3::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    int n = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        n *= inputDesc[0].dims.d[i];

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        // Debug: read input before kernel
        float h_in[4] = {0};
        cudaMemcpyAsync(h_in, inputs[0], sizeof(h_in), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Run kernel
        launch_gelu_fp32(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]), n, stream);

        // Debug: read output after kernel
        float h_out[4] = {0};
        cudaMemcpyAsync(h_out, outputs[0], sizeof(h_out), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        fprintf(stderr, "[GeluV3] n=%d in=[%.4f %.4f %.4f %.4f] out=[%.4f %.4f %.4f %.4f]\n",
            n, h_in[0], h_in[1], h_in[2], h_in[3], h_out[0], h_out[1], h_out[2], h_out[3]);
    } else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
        launch_gelu_fp16(
            static_cast<const __half*>(inputs[0]),
            static_cast<__half*>(outputs[0]), n, stream);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(GeluPluginV3Creator);