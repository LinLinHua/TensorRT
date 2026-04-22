import ctypes
import tensorrt as trt
import onnx
import onnx_graphsurgeon as gs

ctypes.CDLL("/workspace/plugin/build/libgelu_plugin.so", mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

ONNX_PATH = "/workspace/bert_base_gelu.onnx"
PLUGIN_ONNX_PATH = "/workspace/bert_base_bias_gelu.onnx"


@gs.Graph.register()
def replace_with_bias_gelu_plugin(self, inputs, outputs):
    for out in outputs:
        out.inputs.clear()
    return self.layer(op="GeluBiasPluginV3", inputs=inputs, outputs=outputs)


def main():
    print("Loading ONNX graph...")
    graph = gs.import_onnx(onnx.load(ONNX_PATH))

    # Anchor on GeluPluginV3 nodes. If its input comes from an Add(x, bias),
    # fuse both into a single GeluBiasPluginV3.
    gelu_nodes = [n for n in graph.nodes if n.op == "GeluPluginV3"]
    print(f"Found {len(gelu_nodes)} GeluPluginV3 nodes")

    count = 0
    for gelu in gelu_nodes:
        gelu_input = gelu.inputs[0]
        if not gelu_input.inputs:
            continue
        add_node = gelu_input.inputs[0]
        if add_node.op != "Add":
            continue

        actual_x = None
        bias_tensor = None
        for inp in add_node.inputs:
            if isinstance(inp, gs.Constant):
                bias_tensor = inp
            else:
                actual_x = inp
        if actual_x is None or bias_tensor is None:
            continue

        gelu_output = gelu.outputs[0]
        graph.replace_with_bias_gelu_plugin(
            inputs=[actual_x, bias_tensor],
            outputs=[gelu_output],
        )
        count += 1

    print(f"Replaced {count} patterns with GeluBiasPluginV3")
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), PLUGIN_ONNX_PATH)
    print(f"Saved to {PLUGIN_ONNX_PATH}")


if __name__ == "__main__":
    main()