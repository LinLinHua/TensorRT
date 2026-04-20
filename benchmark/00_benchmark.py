import tensorrt as trt
import numpy as np
import torch
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

import ctypes
ctypes.CDLL("/workspace/trt_bert/gelu_plugin/build/libgelu_plugin.so")
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
ENGINE_PATH = "/workspace/trt_bert/bert_fp16_bias_gelu.trt"

def load_engine(path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark(engine, batch_size=1, seq_len=128, n_warmup=20, n_runs=100):
    context = engine.create_execution_context()
    context.set_input_shape("input_ids", (batch_size, seq_len))
    context.set_input_shape("attention_mask", (batch_size, seq_len))

    # Allocate GPU tensors via torch
    input_ids = torch.ones((batch_size, seq_len), dtype=torch.int64).cuda()
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).cuda()

    out0_shape = tuple(context.get_tensor_shape("last_hidden_state"))
    out1_shape = tuple(context.get_tensor_shape("pooler_output"))
    output0 = torch.zeros(out0_shape, dtype=torch.float32).cuda()
    output1 = torch.zeros(out1_shape, dtype=torch.float32).cuda()

    context.set_tensor_address("input_ids", input_ids.data_ptr())
    context.set_tensor_address("attention_mask", attention_mask.data_ptr())
    context.set_tensor_address("last_hidden_state", output0.data_ptr())
    context.set_tensor_address("pooler_output", output1.data_ptr())

    stream = torch.cuda.Stream()

    # Warmup
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.cuda.stream(stream):
        for _ in range(n_runs):
            t0 = time.perf_counter()
            context.execute_async_v3(stream.cuda_stream)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    print(f"Batch={batch_size} | Seq={seq_len} | "
          f"mean={lat.mean():.2f}ms  p50={np.percentile(lat,50):.2f}ms  "
          f"p99={np.percentile(lat,99):.2f}ms | "
          f"Throughput={batch_size*1000/lat.mean():.1f} samples/sec")

if __name__ == "__main__":
    print("Running INT8 benchmark...\n")
    engine = load_engine(ENGINE_PATH)
    print("Running INT8 benchmark...\n")
    for bs in [1, 8, 32]:
        benchmark(engine, batch_size=bs, seq_len=128)