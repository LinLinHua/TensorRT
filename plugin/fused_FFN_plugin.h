#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

void launch_fused_gemm_gelu_fp32(
    const float* A, const float* B, const float* bias, float* C,
    int M, int K, int N, cudaStream_t stream);

class FusedGemmGeluPlugin : public nvinfer1::IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime {
public:
    FusedGemmGeluPlugin() = default;

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

    nvinfer1::IPluginV3* clone() noexcept override {
        return new FusedGemmGeluPlugin();
    }

    const char* getPluginName() const noexcept override { return "FusedGemmGeluPlugin"; }
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
        // Output shape: [batch * seq, N] where N is from weight [K, N]
        outputs[0].nbDims = 2;
        outputs[0].d[0] = inputs[0].d[0]; // M = batch * seq
        outputs[0].d[1] = inputs[1].d[1]; // N from weight
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override {
        return inOut[pos].desc.type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    }

    int32_t configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override {
        return 0;
    }

    size_t getWorkspaceSize(
        const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override {
        return 0;
    }

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

class FusedGemmGeluPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedGemmGeluPluginCreator() { mFC.nbFields = 0; mFC.fields = nullptr; }

    const char* getPluginName() const noexcept override { return "FusedGemmGeluPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    nvinfer1::IPluginV3* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc,
        nvinfer1::TensorRTPhase phase) noexcept override {
        return new FusedGemmGeluPlugin();
    }

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC;
};