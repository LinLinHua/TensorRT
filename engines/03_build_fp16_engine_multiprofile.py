import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ONNX_PATH = "/workspace/bert_base.onnx"
ENGINE_PATH = "/workspace/bert_fp16_multiprofile.trt"

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
config.set_flag(trt.BuilderFlag.FP16)

print("Parsing ONNX...")
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

# Profile 1: optimized for batch=1
profile1 = builder.create_optimization_profile()
profile1.set_shape("input_ids", min=(1,128), opt=(1,128), max=(1,128))
profile1.set_shape("attention_mask", min=(1,128), opt=(1,128), max=(1,128))
config.add_optimization_profile(profile1)

# Profile 2: optimized for batch=8
profile2 = builder.create_optimization_profile()
profile2.set_shape("input_ids", min=(1,128), opt=(8,128), max=(8,128))
profile2.set_shape("attention_mask", min=(1,128), opt=(8,128), max=(8,128))
config.add_optimization_profile(profile2)

# Profile 3: optimized for batch=32
profile3 = builder.create_optimization_profile()
profile3.set_shape("input_ids", min=(1,128), opt=(32,128), max=(32,128))
profile3.set_shape("attention_mask", min=(1,128), opt=(32,128), max=(32,128))
config.add_optimization_profile(profile3)

print("Building multi-profile engine (5-10 min)...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("Engine build failed")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)
print(f"Done! Engine saved to {ENGINE_PATH}")