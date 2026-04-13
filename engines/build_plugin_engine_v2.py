import ctypes
import tensorrt as trt

ctypes.CDLL("/workspace/trt_bert/gelu_plugin/build/libgelu_plugin.so")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

ONNX_PATH = "/workspace/trt_bert/bert_base_plugin.onnx"
ENGINE_PATH = "/workspace/trt_bert/bert_fp16_plugin.trt"

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
config.set_flag(trt.BuilderFlag.FP16)

print("Parsing ONNX with plugin ops...")
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

profile = builder.create_optimization_profile()
profile.set_shape("input_ids", min=(1,128), opt=(8,128), max=(32,128))
profile.set_shape("attention_mask", min=(1,128), opt=(8,128), max=(32,128))
config.add_optimization_profile(profile)

print("Building plugin engine (2-5 min)...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("Engine build failed")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)
print(f"Done! Engine saved to {ENGINE_PATH}")