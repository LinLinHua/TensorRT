import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class GeluBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        return torch.nn.functional.gelu(x + bias)

    @staticmethod
    def symbolic(g, x, bias):
        return g.op("GeluBiasPluginV3", x, bias)

class GeluBiasModule(nn.Module):
    def __init__(self, bias: nn.Parameter):
        super().__init__()
        self.bias = nn.Parameter(bias.data.clone())

    def forward(self, x):
        return GeluBiasFunction.apply(x, self.bias)

def replace_gelu_with_bias_gelu(model):
    from transformers.models.bert.modeling_bert import BertIntermediate
    for name, module in model.named_modules():
        if isinstance(module, BertIntermediate):
            bias = module.dense.bias
            # Remove bias from Linear, handle it in plugin
            module.dense.bias = None
            module.gelu_bias = GeluBiasModule(bias)
            original_act = module.intermediate_act_fn

            def make_forward(gelu_bias_mod):
                def forward(self, x):
                    x = self.dense(x)
                    return gelu_bias_mod(x)
                return forward

            import types
            module.forward = types.MethodType(make_forward(module.gelu_bias), module)

print("Loading BERT-base...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

replace_gelu_with_bias_gelu(model)
print("Bias+GELU replaced with GeluBiasPluginV3")

inputs = tokenizer("Hello, this is a TensorRT test.", return_tensors="pt",
                   padding="max_length", max_length=128)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("Exporting to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "bert_base_bias_gelu.onnx",
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
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
print("Done! bert_base_bias_gelu.onnx saved.")