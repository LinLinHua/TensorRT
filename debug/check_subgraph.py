import onnx
import onnx_graphsurgeon as gs

graph = gs.import_onnx(onnx.load('/workspace/bert_base.onnx'))

for node in graph.nodes:
    if node.op == 'Add' and node.outputs and 'intermediate/dense/Add' in node.outputs[0].name:
        tensor = node.outputs[0]
        print(f'Tensor: {tensor.name}')
        print(f'Number of consumers: {len(tensor.outputs)}')
        for c in tensor.outputs:
            print(f'  → {c.op} name={c.name}')
        break
"
