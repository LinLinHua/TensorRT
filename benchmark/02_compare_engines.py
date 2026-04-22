import ctypes
import tensorrt as trt
import torch

# Load custom plugin .so so TensorRT can recognize all three plugin ops:
# GeluPluginV3, GeluBiasPluginV3, FusedGemmGeluPlugin
ctypes.CDLL("/workspace/plugin/build/libgelu_plugin.so", mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# Engine paths — must match build_engine.py output paths
ENGINE_PATHS = {
    # Standard engines (no custom plugin)
    "baseline_fp32": "/workspace/bert_fp32.trt",
    "baseline_fp16": "/workspace/bert_fp16.trt",
    # GeluPluginV3: GELU only fused
    "gelu_fp32":     "/workspace/bert_gelu_fp32.trt",
    "gelu_fp16":     "/workspace/bert_gelu_fp16.trt",
    # GeluBiasPluginV3: bias + GELU fused
    "bias_gelu_fp32": "/workspace/bert_bias_gelu_fp32.trt",
    "bias_gelu_fp16": "/workspace/bert_bias_gelu_fp16.trt",
    # FusedGemmGeluPlugin: GEMM + bias + GELU fully fused
    "fused_ffn_fp32": "/workspace/bert_fused_ffn_fp32.trt",
    "fused_ffn_fp16": "/workspace/bert_fused_ffn_fp16.trt",
}


def load_engine(path):
    """Deserialize a TensorRT engine from disk."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def run_engine(engine, input_ids, attention_mask):
    """
    Run one forward pass on a TensorRT engine.
    Returns (last_hidden_state, pooler_output) as CPU tensors.
    """
    context = engine.create_execution_context()
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
    context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    return output0.cpu(), output1.cpu()


def try_run(name, path, input_ids, attention_mask):
    """
    Try to load and run an engine.
    Returns (out0, out1) or (None, None) if engine file not found or fails.
    """
    try:
        print(f"  Loading {name}...")
        engine = load_engine(path)
        return run_engine(engine, input_ids, attention_mask)
    except Exception as e:
        print(f"  Skipped {name}: {e}")
        return None, None


def print_table(title, baseline_out0, baseline_out1, engines, input_ids, attention_mask):
    """
    Run all engines and print a comparison table against the baseline.
    """
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Engine':<25} {'last_hidden diff':>16} {'pooler diff':>14}")
    print(f"{'-'*60}")

    sample_outputs = {"Baseline": baseline_out0}

    for name, path in engines:
        out0, out1 = try_run(name, path, input_ids, attention_mask)
        if out0 is None:
            print(f"{name:<25} {'N/A':>16} {'N/A':>14}")
            continue

        diff0 = (baseline_out0 - out0).abs().max().item()
        diff1 = (baseline_out1 - out1).abs().max().item()
        print(f"{name:<25} {diff0:>16.6f} {diff1:>14.6f}")
        sample_outputs[name] = out0

    print(f"{'='*60}")

    # Print sample output values for visual inspection
    print("\nSample outputs (first 5 values of token 0):")
    for name, out in sample_outputs.items():
        print(f"  {name:<20} {out[0, 0, :5]}")

    # Print PASS/WARN status
    print()
    threshold = 0.05
    for name, path in engines:
        out0, _ = try_run(name, path, input_ids, attention_mask)
        if out0 is None:
            continue
        diff = (baseline_out0 - out0).abs().max().item()
        status = "PASS" if diff < threshold else "WARN"
        print(f"  {name}: {status} (diff={diff:.4f})")


if __name__ == "__main__":
    # Test input: batch=1, seq=128, all ones
    input_ids = torch.ones((1, 128), dtype=torch.int64).cuda()
    attention_mask = torch.ones((1, 128), dtype=torch.int64).cuda()

    # ── FP32 comparison ───────────────────────────────────────────
    print("\nLoading FP32 baseline...")
    baseline_fp32 = load_engine(ENGINE_PATHS["baseline_fp32"])
    out0_base_fp32, out1_base_fp32 = run_engine(
        baseline_fp32, input_ids, attention_mask)

    fp32_engines = [
        ("GeluPluginV3 FP32",    ENGINE_PATHS["gelu_fp32"]),
        ("GeluBiasPluginV3 FP32", ENGINE_PATHS["bias_gelu_fp32"]),
        ("FusedGemmGelu FP32",   ENGINE_PATHS["fused_ffn_fp32"]),
    ]

    print_table(
        "FP32 — Plugin vs Baseline",
        out0_base_fp32, out1_base_fp32,
        fp32_engines, input_ids, attention_mask
    )

    # ── FP16 comparison ───────────────────────────────────────────
    print("\nLoading FP16 baseline...")
    baseline_fp16 = load_engine(ENGINE_PATHS["baseline_fp16"])
    out0_base_fp16, out1_base_fp16 = run_engine(
        baseline_fp16, input_ids, attention_mask)

    fp16_engines = [
        ("GeluPluginV3 FP16",    ENGINE_PATHS["gelu_fp16"]),
        ("GeluBiasPluginV3 FP16", ENGINE_PATHS["bias_gelu_fp16"]),
        ("FusedGemmGelu FP16",   ENGINE_PATHS["fused_ffn_fp16"]),
    ]

    print_table(
        "FP16 — Plugin vs Baseline",
        out0_base_fp16, out1_base_fp16,
        fp16_engines, input_ids, attention_mask
    )