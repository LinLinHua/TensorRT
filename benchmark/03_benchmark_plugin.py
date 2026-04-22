import ctypes
import tensorrt as trt
import numpy as np
import torch
import time
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ctypes.CDLL("/workspace/plugin/build/libgelu_plugin.so", mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

ENGINES = {
    "FP32 Baseline":       "/workspace/bert_fp32.trt",
    "FP32 GeluPluginV3":   "/workspace/bert_gelu_fp32.trt",
    "FP32 GeluBias":       "/workspace/bert_bias_gelu_fp32.trt",
    "FP32 FusedFFN":       "/workspace/bert_fused_ffn_fp32.trt",
    "FP16 Baseline":       "/workspace/bert_fp16.trt",
    "FP16 GeluPluginV3":   "/workspace/bert_gelu_fp16.trt",
    "FP16 GeluBias":       "/workspace/bert_bias_gelu_fp16.trt",
    "FP16 FusedFFN":       "/workspace/bert_fused_ffn_fp16.trt",
}

def load_engine(path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark(engine, batch=8, seq=128, n_warmup=20, n_runs=100):
    ctx = engine.create_execution_context()

    ctx.set_input_shape("input_ids", (batch, seq))
    ctx.set_input_shape("attention_mask", (batch, seq))

    ids = torch.ones((batch, seq), dtype=torch.int64).cuda()
    mask = torch.ones((batch, seq), dtype=torch.int64).cuda()

    out0 = torch.zeros(tuple(ctx.get_tensor_shape("last_hidden_state")), dtype=torch.float32).cuda()
    out1 = torch.zeros(tuple(ctx.get_tensor_shape("pooler_output")), dtype=torch.float32).cuda()

    ctx.set_tensor_address("input_ids", ids.data_ptr())
    ctx.set_tensor_address("attention_mask", mask.data_ptr())
    ctx.set_tensor_address("last_hidden_state", out0.data_ptr())
    ctx.set_tensor_address("pooler_output", out1.data_ptr())
    
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    lats = []
    with torch.cuda.stream(stream):
        for _ in range(n_runs):
            t0 = time.perf_counter()
            ctx.execute_async_v3(stream.cuda_stream)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    return np.mean(lats), batch * 1000 / np.mean(lats)

for name, path in ENGINES.items():
    if not os.path.exists(path):
        print(f"Skip {name}: not found")
        continue
    engine = load_engine(path)
    for bs in [1, 8]:
        lat, tput = benchmark(engine, batch=bs)
        print(f"{name:25s} | batch={bs:2d} | {lat:6.2f}ms | {tput:7.1f}/sec")