import tensorrt as trt
import numpy as np
import torch
from transformers import BertTokenizer
from datasets import load_dataset

class BertEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, seq_len=128, batch_size=8, n_samples=512):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cache_file = "/workspace/trt_bert/int8_calib.cache"

        print("Loading SQuAD calibration data...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_dataset("squad", split="train")

        input_ids_list = []
        attention_mask_list = []

        for item in dataset.select(range(n_samples)):
            text = item["question"] + " " + item["context"][:256]
            enc = tokenizer(text, max_length=seq_len, padding="max_length",
                           truncation=True, return_tensors="pt")
            input_ids_list.append(enc["input_ids"])
            attention_mask_list.append(enc["attention_mask"])

        self.input_ids = torch.cat(input_ids_list, dim=0)
        self.attention_mask = torch.cat(attention_mask_list, dim=0)
        self.current_index = 0

        # Pre-allocate GPU buffers
        self.d_input_ids = torch.zeros((batch_size, seq_len),
                                        dtype=torch.int64).cuda()
        self.d_attention_mask = torch.zeros((batch_size, seq_len),
                                             dtype=torch.int64).cuda()
        print(f"Calibrator ready: {n_samples} samples, batch={batch_size}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.input_ids):
            return None
        batch_ids = self.input_ids[self.current_index:
                                   self.current_index + self.batch_size]
        batch_mask = self.attention_mask[self.current_index:
                                         self.current_index + self.batch_size]
        self.d_input_ids.copy_(batch_ids)
        self.d_attention_mask.copy_(batch_mask)
        self.current_index += self.batch_size
        print(f"  Calibration batch {self.current_index // self.batch_size}"
              f"/{len(self.input_ids) // self.batch_size}")
        return [self.d_input_ids.data_ptr(),
                self.d_attention_mask.data_ptr()]

    def read_calibration_cache(self):
        import os
        if os.path.exists(self.cache_file):
            print("Reading calibration cache...")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Calibration cache saved to {self.cache_file}")


if __name__ == "__main__":
    # Build INT8 engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    ONNX_PATH = "/workspace/bert_base.onnx"
    ENGINE_PATH = "/workspace/bert_int8.trt"

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    calibrator = BertEntropyCalibrator(seq_len=128, batch_size=8, n_samples=512)
    config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    profile.set_shape("attention_mask",
        min=(1, 128), opt=(8, 128), max=(32, 128))
    config.add_optimization_profile(profile)

    print("Building INT8 engine (5-10 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"Done! Engine saved to {ENGINE_PATH}")