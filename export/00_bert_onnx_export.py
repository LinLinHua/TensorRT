import torch
from transformers import BertModel, BertTokenizer

print("Loading BERT-base...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

inputs = tokenizer("Hello, this is a TensorRT test.", return_tensors="pt",
                   padding="max_length", max_length=128)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "bert_base.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"},
        "pooler_output": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print("Done! bert_base.onnx saved.")