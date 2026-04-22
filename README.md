# TensorRT-Accelerated BERT Inference Engine

High-performance inference optimization for BERT-base on NVIDIA T4 using TensorRT FP16/INT8 PTQ, custom IPluginV3 plugins with progressive FFN fusion, and multi-profile engine optimization.

## Results

### Precision & Multi-Profile

| Precision | Batch | Latency (mean) | Throughput | vs FP32 |
|-----------|-------|----------------|------------|---------|
| FP32 | 1 | 7.61 ms | 131.4/sec | 1.0× |
| FP32 | 8 | 42.81 ms | 186.9/sec | 1.0× |
| FP32 | 32 | 161.96 ms | 197.6/sec | 1.0× |
| FP16 | 1 | 3.45 ms | 290.3/sec | 2.2× |
| FP16 | 8 | 8.59 ms | 931.2/sec | **5.0×** |
| FP16 | 32 | 34.29 ms | 933.3/sec | 4.7× |
| INT8+FP16 | 1 | 3.34 ms | 299.5/sec | 2.3× |
| INT8+FP16 | 8 | 8.86 ms | 902.8/sec | 4.8× |
| INT8+FP16 | 32 | 35.58 ms | 899.5/sec | 4.6× |
| FP16 Multi-Profile | 1 | **1.98 ms** | **504.8/sec** | **3.8×** |
| FP16 Multi-Profile | 8 | 8.56 ms | 934.6/sec | 5.0× |
| FP16 Multi-Profile | 32 | 34.38 ms | 930.8/sec | 4.7× |

**Key finding:** Multi-profile optimization achieves 27% latency reduction at batch=1 (2.71ms → 1.98ms).

### Custom IPluginV3 — Progressive FFN Fusion Study

Three custom plugins built on the TensorRT 10.x IPluginV3 interface with ONNX-GraphSurgeon graph rewriting, each with deeper fusion scope targeting BERT's FFN sub-layer (`MatMul → Bias → GELU`):

1. **GeluPluginV3** — replaces the 6-node GELU subgraph (Div, Erf, Add, Mul, Mul)
2. **GeluBiasPluginV3** — fuses bias add + GELU into a single kernel pass
3. **FusedGemmGeluPlugin** — fully fuses MatMul + bias + GELU (cuBLAS for GEMM)

**Numerical correctness** (end-to-end max diff vs baseline `last_hidden_state`):

| Plugin | FP32 diff | FP16 diff |
|--------|-----------|-----------|
| GeluPluginV3 | 0.025 | 0.033 |
| GeluBiasPluginV3 | 0.025 | 0.028 |
| FusedGemmGelu | 0.027 | 0.028 |

All passed the <0.05 correctness threshold.

**Latency** (FP16):

| Engine | batch=1 | batch=8 |
|--------|---------|---------|
| Baseline | 1.16 ms | 1.54 ms |
| GeluPluginV3 | 1.44 ms | 1.94 ms |
| GeluBiasPluginV3 | 1.42 ms | 1.98 ms |
| FusedGemmGelu | 1.19 ms | 1.82 ms |

**All plugins were slower than baseline.** This matches NVIDIA's TensorRT-LLM documentation: plugins are intended for fusion patterns TensorRT's pattern-matcher cannot auto-discover (e.g., FlashAttention). GELU and MatMul+Activation are already built-in fusion targets — wrapping them in a plugin creates an opaque boundary that disrupts TensorRT's myelin optimizer from fusing downstream ops (LayerNorm, residual-add), and adds kernel launch overhead.

The plugin work demonstrates correct IPluginV3 implementation and graph rewriting; it does not produce an end-to-end speedup on this workload.

## Environment

- GPU: NVIDIA Tesla T4
- CUDA: 12.6
- TensorRT: 10.3.0
- PyTorch: 2.4.0+cu121
- Python: 3.12
- OS: Ubuntu 22.04

## Project Structure