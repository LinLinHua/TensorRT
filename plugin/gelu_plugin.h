#pragma once //The standard process, avoiding the redifinition
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstdio>

// announce the function we can use
void launch_gelu_fp32(const float* input, float* output, int n, cudaStream_t stream);
void launch_gelu_fp16(const __half* input, __half* output, int n, cudaStream_t stream);

// ─── GeluPluginV3 (single input) ───────────────────────────────────
'''
class PadPlugin :   public IPluginV3, 
                    public IPluginV3OneCore, 
                    public IPluginV3OneBuild, 
                    public IPluginV3OneRuntime
{
    ...override inherited virtual methods.
};
'''
//the template of class (inherent)
class GeluPluginV3 : public nvinfer1::IPluginV3,
                     public nvinfer1::IPluginV3OneCore,
                     public nvinfer1::IPluginV3OneBuild,
                     public nvinfer1::IPluginV3OneRuntime {
public:
    GeluPluginV3() = default;
    '''
    from template
    '''
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        if (type == nvinfer1::PluginCapabilityType::kBUILD)
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        if (type == nvinfer1::PluginCapabilityType::kCORE)
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);

        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override { return new GeluPluginV3(); }//change class name
    const char* getPluginName() const noexcept override { return "GeluPluginV3"; }//chagne name
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    '''
    int32_t PadPlugin::getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
{
    outputTypes[0] = inputTypes[0];
    return 0;
}
    '''
    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override {
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    '''
    int32_t PadPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    we do not change shape
}
    '''
    int32_t getOutputShapes(
        const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        const nvinfer1::DimsExprs* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        outputs[0] = inputs[0];
        return 0;
    }
    '''
    bool PadPlugin::supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
{
    assert(0 <= pos && pos < 2);
    return inOut[pos].desc.format == PluginFormat::kLINEAR && inOut[pos].desc.type == DataType::kFLOAT;
}
    '''
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override {
        assert(0 <= pos && pos < 2);
        return (inOut[pos].desc.type == nvinfer1::DataType::kFLOAT ||
                inOut[pos].desc.type == nvinfer1::DataType::kHALF) && //we have FP32 and FP16
               inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    }

'''
int32_t PadPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override
{
    return 0;
}

int32_t PadPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
{
    return 0;
}
'''
    int32_t configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override { return 0; }

    size_t getWorkspaceSize(
        const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override { return 0; }
'''
int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
{
    // populate outputs and return status code
}
'''
    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;//return 0 in .cpp

    //This plugin does not need configurePlugin and onShapeChange to do anything, so they are no-ops:
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

class GeluPluginV3Creator : public nvinfer1::IPluginCreatorV3One {
public:
    GeluPluginV3Creator() { mFC.nbFields = 0; mFC.fields = nullptr; }
    const char* getPluginName() const noexcept override { return "GeluPluginV3"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
    nvinfer1::IPluginV3* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc,
        nvinfer1::TensorRTPhase phase) noexcept override { return new GeluPluginV3(); }
private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC;
};