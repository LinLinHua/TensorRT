import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Custom GELU that exports as GeluPluginV3 op in ONNX
class GeluPluginV3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.nn.functional.gelu(x)

    @staticmethod
    def symbolic(g, x):
        return g.op("GeluPluginV3", x)

class GeluPluginV3Module(nn.Module):
    def forward(self, x):
        return GeluPluginV3Function.apply(x)

def replace_gelu(model):
    from transformers.activations import GELUActivation, NewGELUActivation, FastGELUActivation
    for name, module in model.named_children():
        if isinstance(module, (GELUActivation, NewGELUActivation, FastGELUActivation, nn.GELU)):
            setattr(model, name, GeluPluginV3Module())
        else:
            replace_gelu(module)

print("Loading BERT-base...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

replace_gelu(model)
print("GELU modules replaced with GeluPluginV3")

inputs = tokenizer("Hello, this is a TensorRT test.", return_tensors="pt",
                   padding="max_length", max_length=128)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("Exporting to ONNX with plugin ops...")
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "bert_base_plugin.onnx",
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
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
)
print("Done! bert_base_plugin.onnx saved.")