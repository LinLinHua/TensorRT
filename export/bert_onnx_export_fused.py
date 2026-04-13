import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import ClassVar, List

class FusedGemmGeluPlugin(nn.Module):
    """Fused Linear + GELU module that exports as FusedGemmGeluPlugin op."""
    
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone())

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return torch.nn.functional.gelu(out)

def replace_linear_gelu(model):
    from transformers.models.bert.modeling_bert import BertIntermediate
    for name, module in model.named_modules():
        if isinstance(module, BertIntermediate):
            fused = FusedGemmGeluPlugin(module.dense)
            module.fused = fused
            module.dense = nn.Identity()
            module.intermediate_act_fn = nn.Identity()

            def make_forward(f):
                def forward(self, x):
                    return f(x)
                return forward

            import types
            module.forward = types.MethodType(make_forward(fused), module)

print("Loading BERT-base...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

replace_linear_gelu(model)
print("Linear + GELU replaced with FusedGemmGeluPlugin")

inputs = tokenizer("Hello, this is a TensorRT test.", return_tensors="pt",
                   padding="max_length", max_length=128)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("Exporting to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "bert_base_fused.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=False,
        export_modules_as_functions={FusedGemmGeluPlugin},
    )
print("Done! bert_base_fused.onnx saved.")