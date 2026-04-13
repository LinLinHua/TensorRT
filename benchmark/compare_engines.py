import ctypes
import tensorrt as trt
import torch

ctypes.CDLL("/workspace/trt_bert/gelu_plugin/build/libgelu_plugin.so")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def load_engine(path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())

def run_engine(engine, input_ids, attention_mask):
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

input_ids = torch.ones((1, 128), dtype=torch.int64).cuda()
attention_mask = torch.ones((1, 128), dtype=torch.int64).cuda()

print("Loading baseline engine...")
baseline_engine = load_engine("/workspace/trt_bert/bert_fp16.trt")
out0_base, out1_base = run_engine(baseline_engine, input_ids, attention_mask)

print("Loading GeluPluginV3 engine...")
plugin_engine = load_engine("/workspace/trt_bert/bert_fp16_plugin.trt")
out0_plugin, out1_plugin = run_engine(plugin_engine, input_ids, attention_mask)

print("Loading GeluBiasPluginV3 engine...")
bias_gelu_engine = load_engine("/workspace/trt_bert/bert_fp16_bias_gelu.trt")
out0_bias, out1_bias = run_engine(bias_gelu_engine, input_ids, attention_mask)

print("Loading FusedGemmGelu engine (experimental)...")
try:
    fused_engine = load_engine("/workspace/trt_bert/bert_fp16_fused.trt")
    out0_fused, out1_fused = run_engine(fused_engine, input_ids, attention_mask)
    fused_diff = (out0_base - out0_fused).abs().max().item()
    fused_available = True
except Exception as e:
    print(f"  Skipped: {e}")
    fused_available = False

diff0 = (out0_base - out0_plugin).abs().max().item()
diff1 = (out1_base - out1_plugin).abs().max().item()
diff2 = (out0_base - out0_bias).abs().max().item()
diff3 = (out1_base - out1_bias).abs().max().item()

print(f"\n{'='*55}")
print(f"{'Engine':<25} {'last_hidden diff':>16} {'pooler diff':>12}")
print(f"{'='*55}")
print(f"{'GeluPluginV3':<25} {diff0:>16.6f} {diff1:>12.6f}")
print(f"{'GeluBiasPluginV3':<25} {diff2:>16.6f} {diff3:>12.6f}")
if fused_available:
    print(f"{'FusedGemmGelu(exp)':<25} {fused_diff:>16.6f} {'N/A':>12}")
print(f"{'='*55}")

print(f"\nBaseline:        {out0_base[0,0,:5]}")
print(f"GeluPlugin:      {out0_plugin[0,0,:5]}")
print(f"GeluBias:        {out0_bias[0,0,:5]}")

for name, diff in [("GeluPluginV3", diff0), ("GeluBiasPluginV3", diff2)]:
    status = "PASS" if diff < 0.05 else "WARN"
    print(f"{name}: {status} (diff={diff:.4f})")