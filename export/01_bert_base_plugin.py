import onnx
import onnx_graphsurgeon as gs

ONNX_PATH = "/workspace/bert_base.onnx"
PLUGIN_ONNX_PATH = "/workspace/bert_base_gelu.onnx"


@gs.Graph.register()
def replace_with_gelu_plugin(self, inputs, outputs):
    # Disconnect output tensor from its original producer
    for out in outputs:
        out.inputs.clear()
    # Insert new op producing the same output
    return self.layer(op="GeluPluginV3", inputs=inputs, outputs=outputs)


def main():
    graph = gs.import_onnx(onnx.load(ONNX_PATH))
    tmap = graph.tensors()

    # Anchor on each Erf node - one per GELU in BERT
    erfs = [n for n in graph.nodes if n.op == "Erf"]
    print(f"Found {len(erfs)} Erf nodes")

    for erf in erfs:
        # Walk up to find GELU input x
        div = erf.i()
        x = next(inp for inp in div.inputs if not isinstance(inp, gs.Constant))

        # Walk down to find GELU final output
        # Erf -> Add -> (Mul const) -> final Mul
        node = erf.o()
        while node.o().op in ("Add", "Mul"):
            node = node.o()
        gelu_out = node.outputs[0]

        graph.replace_with_gelu_plugin(inputs=[x], outputs=[gelu_out])

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), PLUGIN_ONNX_PATH)
    print(f"Saved {PLUGIN_ONNX_PATH}")


if __name__ == "__main__":
    main()