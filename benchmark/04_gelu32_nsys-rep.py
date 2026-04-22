import ctypes, tensorrt as trt, torch

#init register for plugin
ctypes.CDLL('/workspace/plugin/build/libgelu_plugin.so', mode=ctypes.RTLD_GLOBAL)
L = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(L, '')

engine = trt.Runtime(L).deserialize_cuda_engine(open('/workspace/bert_gelu_fp32.trt','rb').read())
ctx = engine.create_execution_context()
#dummy input
ids = torch.ones((8,128), dtype=torch.int64).cuda()
mask = torch.ones((8,128), dtype=torch.int64).cuda()
ctx.set_input_shape('input_ids', ids.shape)
ctx.set_input_shape('attention_mask', mask.shape)
#define the GPU address (to tell tensorRT)
out0 = torch.zeros(tuple(ctx.get_tensor_shape('last_hidden_state')), dtype=torch.float32).cuda()
out1 = torch.zeros(tuple(ctx.get_tensor_shape('pooler_output')), dtype=torch.float32).cuda()
ctx.set_tensor_address('input_ids', ids.data_ptr())
ctx.set_tensor_address('attention_mask', mask.data_ptr())
ctx.set_tensor_address('last_hidden_state', out0.data_ptr())
ctx.set_tensor_address('pooler_output', out1.data_ptr())

# warmup
for _ in range(10):
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()

# actual runs (nsys records these)
for _ in range(30):
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()