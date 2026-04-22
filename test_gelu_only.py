#!/usr/bin/env python3
"""
Gelu-only plugin test: build engine + compare vs baseline.
Run `make plugin` externally first.
"""
import ctypes
import tensorrt as trt
import torch

PLUGIN_SO = "/workspace/plugin/build/libgelu_plugin.so"
ONNX_PATH = "/workspace/bert_base_gelu.onnx"
ENGINE_PATH = "/workspace/bert_gelu_fp32.trt"
BASELINE_ENGINE = "/workspace/bert_fp32.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_plugin():
    ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def build_engine():
    print(f"Building {ENGINE_PATH} from {ONNX_PATH}...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", min=(1, 128), opt=(8, 128), max=(32, 128))
    profile.set_shape("attention_mask", min=(1, 128), opt=(8, 128), max=(32, 128))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"Engine saved: {ENGINE_PATH}\n")


def run_engine(path, ids, mask):
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(open(path, "rb").read())
    ctx = engine.create_execution_context()
    ctx.set_input_shape("input_ids", ids.shape)
    ctx.set_input_shape("attention_mask", mask.shape)

    out0 = torch.zeros(tuple(ctx.get_tensor_shape("last_hidden_state")), dtype=torch.float32).cuda()
    out1 = torch.zeros(tuple(ctx.get_tensor_shape("pooler_output")), dtype=torch.float32).cuda()

    ctx.set_tensor_address("input_ids", ids.data_ptr())
    ctx.set_tensor_address("attention_mask", mask.data_ptr())
    ctx.set_tensor_address("last_hidden_state", out0.data_ptr())
    ctx.set_tensor_address("pooler_output", out1.data_ptr())

    stream = torch.cuda.Stream()
    ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    return out0.cpu()


def compare():
    ids = torch.ones((1, 128), dtype=torch.int64).cuda()
    mask = torch.ones((1, 128), dtype=torch.int64).cuda()

    base = run_engine(BASELINE_ENGINE, ids, mask)
    plugin = run_engine(ENGINE_PATH, ids, mask)

    diff = (base - plugin).abs().max().item()
    print(f"GeluPluginV3 FP32 diff: {diff:.6f}")
    print("PASS" if diff < 0.05 else "FAIL")


if __name__ == "__main__":
    import os
    load_plugin()
    if os.path.exists(ENGINE_PATH):
        os.remove(ENGINE_PATH)
    build_engine()
    compare()