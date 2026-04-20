#include "gelu_plugin.h"
#include <cstdio>

// ─── GeluPluginV3 enqueue ─────────────────────────────────────────────────────

int32_t GeluPluginV3::enqueue(
    //copy from template
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    '''
    inputDes=input type, shape (Desc=Descriper)
    outputDesc=output type, shape
    '''
    int n = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        n *= inputDesc[0].dims.d[i];//inputs[0] >GELU (1 input)

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
