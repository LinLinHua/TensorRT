import ctypes
import tensorrt as trt
import torch

ctypes.CDLL("/workspace/plugin/build/libgelu_plugin.so", mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def test_plugin(plugin_name, input_shape, num_inputs=1):
    """
    Test a single plugin by building a minimal engine and comparing output to PyTorch GELU.
    """
    registry = trt.get_plugin_registry()
    creator = registry.get_creator(plugin_name, "1", "")
    if creator is None:
        print(f"[{plugin_name}] FAIL - creator not found in registry")
        return

    fc = trt.PluginFieldCollection()
    plugin = creator.create_plugin(plugin_name, fc, trt.TensorRTPhase.BUILD)
    print(f"[{plugin_name}] Plugin created: {plugin}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()

    # Add inputs
    inputs = []
    inp0 = network.add_input("input", trt.DataType.FLOAT, tuple(-1 if i == 0 else d for i, d in enumerate(input_shape)))
    inputs.append(inp0)

    gelu_layer = network.add_plugin_v3(inputs, [], plugin)
    network.mark_output(gelu_layer.get_output(0))

    profile = builder.create_optimization_profile()
    profile.set_shape("input",
        min=tuple(1 if i == 0 else d for i, d in enumerate(input_shape)),
        opt=tuple(input_shape),
        max=tuple(input_shape))
    config.add_optimization_profile(profile)

    print(f"[{plugin_name}] Building engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print(f"[{plugin_name}] FAIL - engine build failed")
        return

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    context.set_input_shape("input", tuple(input_shape))

    x = torch.randn(*input_shape, dtype=torch.float32).cuda()
    output_name = engine.get_tensor_name(1)
    output_trt = torch.zeros(*input_shape, dtype=torch.float32).cuda()

    context.set_tensor_address("input", x.data_ptr())
    context.set_tensor_address(output_name, output_trt.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    output_ref = torch.nn.functional.gelu(x)
    max_diff = (output_trt - output_ref).abs().max().item()
    status = "PASS ✓" if max_diff < 1e-2 else "FAIL ✗"
    print(f"[{plugin_name}] Max diff vs PyTorch GELU: {max_diff:.6f} — {status}\n")


if __name__ == "__main__":
    # Test GeluPluginV3 (single input: x)
    test_plugin("GeluPluginV3", input_shape=[1, 128, 3072])

    # GeluBiasPluginV3 and FusedGemmGeluPlugin need multiple inputs
    # which requires a different setup — test separately below
    print("Note: GeluBiasPluginV3 and FusedGemmGeluPlugin require")
    print("multiple inputs and cannot be tested with this simple harness.")
    print("Use compare_engines.py for full end-to-end validation.")