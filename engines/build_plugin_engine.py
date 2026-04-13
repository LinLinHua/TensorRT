import ctypes
import tensorrt as trt
import onnx
import onnx_graphsurgeon as gs
import numpy as np

ctypes.CDLL("/workspace/trt_bert/gelu_plugin/build/libgelu_plugin.so")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

ONNX_PATH = "/workspace/trt_bert/bert_base.onnx"
PLUGIN_ONNX_PATH = "/workspace/trt_bert/bert_base_plugin.onnx"
ENGINE_PATH = "/workspace/trt_bert/bert_fp16_plugin.trt"

def find_gelu_subgraphs(graph):
    """
    Find Mul -> Erf -> Add -> Mul -> Mul pattern (GELU decomposition)
    Returns list of (input_tensor, output_tensor, nodes_to_remove)
    """
    gelu_subgraphs = []
    
    # Build a map of output tensor -> node
    output_to_node = {}
    for node in graph.nodes:
        for out in node.outputs:
            output_to_node[out.name] = node

    visited = set()
    
    for node in graph.nodes:
        # Look for the final Mul in GELU pattern
        if node.op != "Mul" or id(node) in visited:
            continue
        
        # Pattern: x * 0.5 * (1 + erf(x / sqrt(2)))
        # Final Mul: mul(x, mul(0.5, add(1, erf(div(x, sqrt(2))))))
        # Or: mul(mul(x, 0.5), add(1, erf(...)))
        
        try:
            # Check if one input comes from an Add node
            add_node = None
            x_input = None
            
            for inp in node.inputs:
                if hasattr(inp, 'inputs') and inp.name in output_to_node:
                    parent = output_to_node[inp.name]
                    if parent.op == "Add":
                        add_node = parent
                    else:
                        x_input = inp
                else:
                    x_input = inp
            
            if add_node is None:
                continue
                
            # Check Add has Erf as one input
            erf_node = None
            for inp in add_node.inputs:
                if inp.name in output_to_node:
                    parent = output_to_node[inp.name]
                    if parent.op == "Erf":
                        erf_node = parent
                        break
            
            if erf_node is None:
                continue
            
            # Check Erf input comes from Div or Mul (x/sqrt(2))
            div_or_mul = None
            for inp in erf_node.inputs:
                if inp.name in output_to_node:
                    parent = output_to_node[inp.name]
                    if parent.op in ["Div", "Mul"]:
                        div_or_mul = parent
                        break
            
            if div_or_mul is None:
                continue
            
            # Found GELU pattern
            # Get the original input x (first input to div_or_mul that's not a constant)
            gelu_input = None
            for inp in div_or_mul.inputs:
                if not isinstance(inp, gs.Constant):
                    gelu_input = inp
                    break
            
            if gelu_input is None:
                continue
            
            gelu_output = node.outputs[0]
            nodes_to_remove = [node, add_node, erf_node, div_or_mul]
            
            gelu_subgraphs.append((gelu_input, gelu_output, nodes_to_remove))
            visited.add(id(node))
            
        except Exception as e:
            continue
    
    return gelu_subgraphs

def replace_gelu_with_plugin():
    print("Loading ONNX graph...")
    graph = gs.import_onnx(onnx.load(ONNX_PATH))
    
    subgraphs = find_gelu_subgraphs(graph)
    print(f"Found {len(subgraphs)} GELU subgraphs")

    for i, (gelu_input, gelu_output, nodes_to_remove) in enumerate(subgraphs):
        # Use graph.layer() — NVIDIA's official pattern from example 08
        # Step 1: Disconnect input tensor outputs (severs link to old subgraph)
        gelu_input.outputs.clear()
        # Step 2: Disconnect output tensor inputs (severs link from old subgraph)
        gelu_output.inputs.clear()
        # Step 3: Insert plugin node using graph.layer()
        graph.layer(op="GeluPluginV3", name=f"gelu_plugin_{i}",
                    inputs=[gelu_input], outputs=[gelu_output])

    print(f"Replaced {len(subgraphs)} GELU subgraphs")
    
    # Let cleanup() automatically remove now-dangling old nodes
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), PLUGIN_ONNX_PATH)
    print(f"Saved modified ONNX to {PLUGIN_ONNX_PATH}")

def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    print("Parsing modified ONNX...")
    with open(PLUGIN_ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    profile.set_shape("attention_mask",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    config.add_optimization_profile(profile)

    print("Building plugin engine (2-5 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"Done! Engine saved to {ENGINE_PATH}")

if __name__ == "__main__":
    replace_gelu_with_plugin()
    build_engine()