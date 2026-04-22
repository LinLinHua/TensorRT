# Top-level Makefile for TensorRT BERT inference pipeline

WORKSPACE = /workspace
PLUGIN_DIR = $(WORKSPACE)/plugin
BUILD_DIR  = $(PLUGIN_DIR)/build
IMAGE_NAME = trt_bert
CONTAINER_NAME = trt_bert_container

# # ── Docker ──────────────────────────────────────────────────────

# # Build Docker image from Dockerfile
# .PHONY: docker-build
# docker-build:
# 	docker build -t $(IMAGE_NAME):latest .

# # Push image to Docker Hub (set DOCKERHUB_USER before running)
# # Usage: make docker-push DOCKERHUB_USER=yourname
# .PHONY: docker-push
# docker-push:
# 	docker tag $(IMAGE_NAME):latest $(DOCKERHUB_USER)/$(IMAGE_NAME):latest
# 	docker push $(DOCKERHUB_USER)/$(IMAGE_NAME):latest

# # Run container with GPU access and mount current directory
# .PHONY: docker-run
# docker-run:
# 	docker run --gpus all -it \
# 		--name $(CONTAINER_NAME) \
# 		-v $(PWD):/workspace/ \
# 		$(IMAGE_NAME):latest

# # Remove stopped container
# .PHONY: docker-clean
# docker-clean:
# 	docker rm -f $(CONTAINER_NAME)

# ── Plugin ──────────────────────────────────────────────────────
.PHONY: plugin
plugin:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make -j4

# ── Export ──────────────────────────────────────────────────────
.PHONY: export
export:
	cd $(WORKSPACE) && python3 export/00_bert_onnx_export.py
	cd $(WORKSPACE) && python3 export/01_bert_base_plugin.py
	cd $(WORKSPACE) && python3 export/02_bert_onnx_export_bias_gelu.py
	cd $(WORKSPACE) && python3 export/03_bert_onnx_export_fused.py

# ── Build all TensorRT engines ──────────────────────────────────
.PHONY: build
build:
	cd $(WORKSPACE) && python3 engines/00_build_fp32_engine.py
	cd $(WORKSPACE) && python3 engines/01_build_fp16_engine.py
# 	cd $(WORKSPACE) && python3 engines/02_int8_calibrator_and_engine.py
# 	cd $(WORKSPACE) && python3 engines/03_build_fp16_engine_multiprofile.py
	cd $(WORKSPACE) && python3 engines/04_build_plugin_engine.py
	

# ── Compare engines ─────────────────────────────────────────────
.PHONY: compare
compare:
# 	cd $(WORKSPACE) && python3 benchmark/00_benchmark.py
# 	cd $(WORKSPACE) && python3 benchmark/01_benchmark_multiprofile.py
	cd $(WORKSPACE) && python3 benchmark/02_compare_engines.py



# ── Full pipeline ───────────────────────────────────────────────
.PHONY: all
all: plugin export build compare

# ── Clean build artifacts ────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(WORKSPACE)/*.onnx
	rm -f $(WORKSPACE)/*.trt