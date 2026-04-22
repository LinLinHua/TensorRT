import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = "/workspace/bert_base.onnx"
ENGINE_PATH = "/workspace/bert_fp16.trt"

def build_fp16_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("Parsing ONNX...")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Enable FP16
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    profile.set_shape("attention_mask",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    config.add_optimization_profile(profile)

    print("Building FP16 engine (2-5 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"Done! Engine saved to {ENGINE_PATH}")

if __name__ == "__main__":
    build_fp16_engine()