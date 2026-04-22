import ctypes
import tensorrt as trt

# Load custom plugin .so so TensorRT can recognize all three plugin ops:
# GeluPluginV3, GeluBiasPluginV3, FusedGemmGeluPlugin
ctypes.CDLL("/workspace/plugin/build/libgelu_plugin.so", mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# ONNX paths (outputs from the three prepare_*.py scripts)
ONNX_PATHS = {
    "gelu":       "/workspace/bert_base_gelu.onnx",
    "bias_gelu":  "/workspace/bert_base_bias_gelu.onnx",
    "fused_ffn":  "/workspace/bert_base_fused_ffn.onnx",
}

# Engine output paths
ENGINE_PATHS = {
    "gelu_fp32":      "/workspace/bert_gelu_fp32.trt",
    "gelu_fp16":      "/workspace/bert_gelu_fp16.trt",
    "bias_gelu_fp32": "/workspace/bert_bias_gelu_fp32.trt",
    "bias_gelu_fp16": "/workspace/bert_bias_gelu_fp16.trt",
    "fused_ffn_fp32": "/workspace/bert_fused_ffn_fp32.trt",
    "fused_ffn_fp16": "/workspace/bert_fused_ffn_fp16.trt",
}


def build_engine(onnx_path: str, engine_path: str, fp16: bool):
    """
    Build a TensorRT engine from a plugin ONNX file.
    fp16=True enables FP16 precision, fp16=False builds FP32 engine.
    """
    precision = "FP16" if fp16 else "FP32"

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 4GB workspace for TensorRT kernel auto-tuning
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, False)  
    if fp16:
        # Allow TensorRT to use FP16 kernels where beneficial
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Parsing {onnx_path} for {precision} engine...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    # Dynamic shape profile: batch size 1-32, sequence length fixed at 128
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    profile.set_shape("attention_mask",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    config.add_optimization_profile(profile)

    print(f"Building {precision} engine (2-5 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Done! Engine saved to {engine_path}")


if __name__ == "__main__":
    # Build FP32 and FP16 engines for all three plugin variants
    build_engine(ONNX_PATHS["gelu"],      ENGINE_PATHS["gelu_fp32"],      fp16=False)
    build_engine(ONNX_PATHS["gelu"],      ENGINE_PATHS["gelu_fp16"],      fp16=True)
    build_engine(ONNX_PATHS["bias_gelu"], ENGINE_PATHS["bias_gelu_fp32"], fp16=False)
    build_engine(ONNX_PATHS["bias_gelu"], ENGINE_PATHS["bias_gelu_fp16"], fp16=True)
    build_engine(ONNX_PATHS["fused_ffn"], ENGINE_PATHS["fused_ffn_fp32"], fp16=False)
    build_engine(ONNX_PATHS["fused_ffn"], ENGINE_PATHS["fused_ffn_fp16"], fp16=True)