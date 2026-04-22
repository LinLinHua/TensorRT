# Base image: NVIDIA TensorRT with Python 3, includes CUDA, cuBLAS, and TensorRT pre-installed
FROM nvcr.io/nvidia/tensorrt:24.01-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages required for BERT export and TensorRT pipeline
RUN pip install --no-cache-dir \
    torch \
    transformers \
    onnx \
    onnxruntime-gpu \
    onnx-graphsurgeon \
    polygraphy \
    datasets

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]