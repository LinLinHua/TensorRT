#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstdio>

// Fused bias + GELU
void launch_gelu_bias_fp32(const float* input, const float* bias, float* output,
    int n, int hidden_size, cudaStream_t stream);
void launch_gelu_bias_fp16(const __half* input, const __half* bias, __half* output,
    int n, int hidden_size, cudaStream_t stream);


// ─── GeluBiasPluginV3 (fused bias + GELU, two inputs) ────────────────────────

class GeluBiasPluginV3 : public nvinfer1::IPluginV3,
                          public nvinfer1::IPluginV3OneCore,
                          public nvinfer1::IPluginV3OneBuild,
                          public nvinfer1::IPluginV3OneRuntime {
public:
    GeluBiasPluginV3() = default;

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override {
        if (type == nvinfer1::PluginCapabilityType::kCORE)
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        if (type == nvinfer1::PluginCapabilityType::kBUILD)
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override { return new GeluBiasPluginV3(); }
    const char* getPluginName() const noexcept override { return "GeluBiasPluginV3"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override {
        outputTypes[0] = nvinfer1::DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(
        const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        const nvinfer1::DimsExprs* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        outputs[0] = inputs[0];  // Output same shape as input x
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override {
        return (inOut[pos].desc.type == nvinfer1::DataType::kFLOAT ||
                inOut[pos].desc.type == nvinfer1::DataType::kHALF)&&
               inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    }

    int32_t configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override { return 0; }

    size_t getWorkspaceSize(
        const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override { return 0; }

    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(
        const nvinfer1::PluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* out, int32_t nbOutputs) noexcept override { return 0; }

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override { return clone(); }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFC; }

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{0, nullptr};
};

class GeluBiasPluginV3Creator : public nvinfer1::IPluginCreatorV3One {
public:
    GeluBiasPluginV3Creator() { mFC.nbFields = 0; mFC.fields = nullptr; }
    const char* getPluginName() const noexcept override { return "GeluBiasPluginV3"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
    nvinfer1::IPluginV3* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc,
        nvinfer1::TensorRTPhase phase) noexcept override { return new GeluBiasPluginV3(); }
private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC;
};