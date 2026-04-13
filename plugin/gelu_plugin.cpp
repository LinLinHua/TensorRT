#include "gelu_plugin.h"
#include <cstdio>

// ─── GeluPluginV3 enqueue ─────────────────────────────────────────────────────

int32_t GeluPluginV3::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    int n = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        n *= inputDesc[0].dims.d[i];

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        launch_gelu_fp32(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]), n, stream);
    } else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
        launch_gelu_fp16(
            static_cast<const __half*>(inputs[0]),
            static_cast<__half*>(outputs[0]), n, stream);
    }
    return 0;
}

// ─── GeluBiasPluginV3 enqueue ─────────────────────────────────────────────────

int32_t GeluBiasPluginV3::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    // inputs[0]: x [batch, seq, hidden_size]
    // inputs[1]: bias [hidden_size]
    int n = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        n *= inputDesc[0].dims.d[i];
    int hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        launch_gelu_bias_fp32(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<float*>(outputs[0]),
            n, hidden_size, stream);
    } else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
        launch_gelu_bias_fp16(
            static_cast<const __half*>(inputs[0]),
            static_cast<const __half*>(inputs[1]),
            static_cast<__half*>(outputs[0]),
            n, hidden_size, stream);
    }
    return 0;
}

// ─── Plugin registration ──────────────────────────────────────────────────────

__attribute__((constructor)) static void registerPlugins() {
    static GeluPluginV3Creator geluCreator;
    static GeluBiasPluginV3Creator geluBiasCreator;
    fprintf(stderr, "[GeluPlugin] Registering GeluPluginV3...\n");
    getPluginRegistry()->registerCreator(geluCreator, "");
    fprintf(stderr, "[GeluPlugin] Registering GeluBiasPluginV3...\n");
    getPluginRegistry()->registerCreator(geluBiasCreator, "");
    fprintf(stderr, "[GeluPlugin] Done.\n");
}