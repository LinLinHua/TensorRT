import ctypes
import tensorrt as trt
import onnx
import onnx_graphsurgeon as gs

ctypes.CDLL("/workspace/plugin/build/libgelu_plugin.so", mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

ONNX_PATH = "/workspace/bert_base_bias_gelu.onnx"
PLUGIN_ONNX_PATH = "/workspace/bert_base_fused_ffn.onnx"


@gs.Graph.register()
def replace_with_fused_ffn_plugin(self, inputs, outputs):
    for out in outputs:
        out.inputs.clear()
    return self.layer(op="FusedGemmGeluPlugin", inputs=inputs, outputs=outputs)


def main():
    print("Loading ONNX graph...")
    graph = gs.import_onnx(onnx.load(ONNX_PATH))

    # Anchor on GeluBiasPluginV3. If its first input comes from a MatMul,
    # fuse MatMul + bias + GELU into a single FusedGemmGeluPlugin.
    bias_gelu_nodes = [n for n in graph.nodes if n.op == "GeluBiasPluginV3"]
    print(f"Found {len(bias_gelu_nodes)} GeluBiasPluginV3 nodes")

    count = 0
    for bg in bias_gelu_nodes:
        if len(bg.inputs) < 2:
            continue
        gemm_output = bg.inputs[0]
        bias_tensor = bg.inputs[1]

        if not gemm_output.inputs:
            continue
        matmul = gemm_output.inputs[0]
        if matmul.op != "MatMul":
            continue

        actual_x = None
        weight = None
        for inp in matmul.inputs:
            if isinstance(inp, gs.Constant):
                weight = inp
            else:
                actual_x = inp
        if actual_x is None or weight is None:
            continue

        ffn_output = bg.outputs[0]
        graph.replace_with_fused_ffn_plugin(
            inputs=[actual_x, weight, bias_tensor],
            outputs=[ffn_output],
        )
        count += 1

    print(f"Replaced {count} patterns with FusedGemmGeluPlugin")
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), PLUGIN_ONNX_PATH)
    print(f"Saved to {PLUGIN_ONNX_PATH}")


if __name__ == "__main__":
    main()