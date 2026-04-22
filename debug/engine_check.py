#!/usr/bin/env python3
"""
Debug: inspect engine layers to verify GeluPluginV3 usage.
Saves output to /workspace/debug/engine_log.txt
"""
import ctypes
import json
import os
import tensorrt as trt

PLUGIN_SO = "/workspace/plugin/build/libgelu_plugin.so"
ENGINE_PATH = "/workspace/bert_gelu_fp32.trt"
OUTPUT_DIR = "/workspace/debug"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "engine_log.txt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(
        open(ENGINE_PATH, "rb").read()
    )
    inspector = engine.create_engine_inspector()
    info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    info = json.loads(info_str)
    layers = info.get("Layers", [])

    # Normalize: some TRT versions return list of strings, others return list of dicts
    parsed_layers = []
    for l in layers:
        if isinstance(l, str):
            try:
                parsed_layers.append(json.loads(l))
            except json.JSONDecodeError:
                parsed_layers.append({"Name": l, "LayerType": "Unknown"})
        elif isinstance(l, dict):
            parsed_layers.append(l)

    summary_lines = []
    summary_lines.append(f"Engine: {ENGINE_PATH}")
    summary_lines.append(f"Total layers: {len(parsed_layers)}")

    # Count by layer type
    type_counts = {}
    for l in parsed_layers:
        t = l.get("LayerType", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    summary_lines.append("\n--- Layer type counts ---")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {t}: {c}")

    # Gelu-related layers
    gelu_layers = [l for l in parsed_layers if "elu" in str(l).lower()]
    summary_lines.append(f"\n--- Gelu-related layers ({len(gelu_layers)}) ---")
    for l in gelu_layers:
        name = l.get("Name", "?")
        ltype = l.get("LayerType", "?")
        origin = l.get("Origin", "?")
        summary_lines.append(f"  [{ltype}] {name}  (Origin: {origin})")

    # Plugin layers
    plugin_layers = [l for l in parsed_layers if "Plugin" in l.get("LayerType", "")]
    summary_lines.append(f"\n--- Plugin layers ({len(plugin_layers)}) ---")
    for l in plugin_layers:
        summary_lines.append(f"  [{l.get('LayerType','?')}] {l.get('Name','?')}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(OUTPUT_FILE, "w") as f:
        f.write(summary_text)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("FULL ENGINE INFORMATION (JSON)\n")
        f.write("=" * 70 + "\n\n")
        f.write(json.dumps(info, indent=2))

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()