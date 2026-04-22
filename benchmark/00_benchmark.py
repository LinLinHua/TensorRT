import tensorrt as trt
import numpy as np
import torch
import time
import ctypes
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ctypes.CDLL("/workspace/plugin/build/libgelu_plugin.so", mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

ENGINE_PATHS = {
    "FP32":              "/workspace/bert_fp32.trt",
    "FP16":              "/workspace/bert_fp16.trt",
    "INT8+FP16":         "/workspace/bert_int8.trt",
    "FP16 Multi-Profile": "/workspace/bert_fp16_multiprofile.trt",
}

def load_engine(path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark(engine, batch_size=1, seq_len=128, n_warmup=20, n_runs=100, profile_idx=0):
    context = engine.create_execution_context()
    context.set_optimization_profile_async(profile_idx, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    context.set_input_shape("input_ids", (batch_size, seq_len))
    context.set_input_shape("attention_mask", (batch_size, seq_len))

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

    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    latencies = []
    with torch.cuda.stream(stream):
        for _ in range(n_runs):
            t0 = time.perf_counter()
            context.execute_async_v3(stream.cuda_stream)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    return {
        "batch": batch_size,
        "mean": lat.mean(),
        "p50": np.percentile(lat, 50),
        "p99": np.percentile(lat, 99),
        "throughput": batch_size * 1000 / lat.mean(),
    }

if __name__ == "__main__":
    os.makedirs("/workspace/results", exist_ok=True)
    lines = []

    for precision, path in ENGINE_PATHS.items():
        if not os.path.exists(path):
            print(f"Skipping {precision}: {path} not found")
            continue

        print(f"\nBenchmarking {precision}...")
        engine = load_engine(path)

        # Multi-profile uses 3 profiles (batch 1, 8, 32)
        profile_map = {1: 0, 8: 1, 32: 2} if "Multi-Profile" in precision else {1: 0, 8: 0, 32: 0}

        for bs in [1, 8, 32]:
            r = benchmark(engine, batch_size=bs, profile_idx=profile_map[bs])
            line = (f"{precision} | Batch={r['batch']} | "
                    f"mean={r['mean']:.2f}ms  p50={r['p50']:.2f}ms  "
                    f"p99={r['p99']:.2f}ms | "
                    f"Throughput={r['throughput']:.1f} samples/sec")
            print(line)
            lines.append(line)

    with open("/workspace/results/00_quantization.txt", "w") as f:
        f.write("\n".join(lines))
    print("\nSaved to /workspace/results/00_quantization.txt")