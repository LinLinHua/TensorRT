import tensorrt as trt
import torch
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = "/workspace/trt_bert/bert_fp16_multiprofile.trt"

def benchmark(engine, batch_size, seq_len=128, profile_idx=0, n_runs=200, warmup=50):
    context = engine.create_execution_context()
    context.set_optimization_profile_async(profile_idx, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    input_ids = torch.ones((batch_size, seq_len), dtype=torch.int64).cuda()
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).cuda()

    context.set_input_shape("input_ids", input_ids.shape)
    context.set_input_shape("attention_mask", attention_mask.shape)

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
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    mean_ms = np.mean(latencies)
    p50_ms = np.percentile(latencies, 50)
    p99_ms = np.percentile(latencies, 99)
    throughput = batch_size / (mean_ms / 1000)

    print(f"Batch={batch_size} | Profile={profile_idx} | mean={mean_ms:.2f}ms "
          f"p50={p50_ms:.2f}ms p99={p99_ms:.2f}ms | Throughput={throughput:.1f} samples/sec")

runtime = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

print("Multi-profile benchmark:")
benchmark(engine, batch_size=1,  profile_idx=0)
benchmark(engine, batch_size=8,  profile_idx=1)
benchmark(engine, batch_size=32, profile_idx=2)