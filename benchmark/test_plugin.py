import ctypes
import tensorrt as trt
import torch

ctypes.CDLL("/workspace/trt_bert/gelu_plugin/build/libgelu_plugin.so")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

registry = trt.get_plugin_registry()
creator = registry.get_creator("GeluPluginV3", "1", "")
fc = trt.PluginFieldCollection()
plugin = creator.create_plugin("gelu", fc, trt.TensorRTPhase.BUILD)
print(f"Plugin created: {plugin}")

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
config = builder.create_builder_config()

inp = network.add_input("input", trt.DataType.FLOAT, (-1, 128, 3072))
gelu_layer = network.add_plugin_v3([inp], [], plugin)
network.mark_output(gelu_layer.get_output(0))

profile = builder.create_optimization_profile()
profile.set_shape("input", min=(1,128,3072), opt=(8,128,3072), max=(32,128,3072))
config.add_optimization_profile(profile)

print("Building engine...")
serialized = builder.build_serialized_network(network, config)
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized)

context = engine.create_execution_context()
context.set_input_shape("input", (1, 128, 3072))

x = torch.randn(1, 128, 3072, dtype=torch.float32).cuda()
output_name = engine.get_tensor_name(1)
output_trt = torch.zeros(1, 128, 3072, dtype=torch.float32).cuda()

context.set_tensor_address("input", x.data_ptr())
context.set_tensor_address(output_name, output_trt.data_ptr())

stream = torch.cuda.Stream()
context.execute_async_v3(stream.cuda_stream)
torch.cuda.synchronize()

output_ref = torch.nn.functional.gelu(x)
max_diff = (output_trt - output_ref).abs().max().item()
print(f"Max diff vs PyTorch GELU: {max_diff:.6f}")
print("PASS ✓" if max_diff < 1e-3 else "FAIL")