import ctypes
import tensorrt as trt

ctypes.CDLL("/workspace/trt_bert/gelu_plugin/build/libgelu_plugin.so")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

registry = trt.get_plugin_registry()

# Try V3 creator
creator_v3 = registry.get_creator("GeluPluginV3", "1", "")
print(f"V3 creator: {creator_v3}")

# Try creating plugin via V3 path
if creator_v3:
    fc = trt.PluginFieldCollection()
    plugin = creator_v3.create_plugin("gelu", fc, trt.TensorRTPhase.BUILD)
    print(f"Plugin: {plugin}")
    print(f"Plugin type: {type(plugin)}")