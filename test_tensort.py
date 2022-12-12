import numpy as np
import torch
import tensorrt as trt
import common
from datetime import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

engine_model_path = "distillbert_seq_128.plan"
# Build a TensorRT engine.
engine = get_engine(engine_model_path)
# Contexts are used to perform inference.
context = engine.create_execution_context()


"""
b、从engine中获取inputs, outputs, bindings, stream 的格式以及分配缓存
"""
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

input_seq=128

sentence = ["i hate Shake. He is a bad guy in the baskerball game"]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
inputs = tokenizer.batch_encode_plus(sentence, return_tensors='pt', add_special_tokens=True)


tokens_id =  to_numpy(inputs['input_ids'].int())
attention_mask = to_numpy(inputs['attention_mask'].int())

raw_seq=tokens_id.shape[-1]
batchsize=tokens_id.shape[0]

tokens_id=np.hstack((tokens_id,np.zeros((batchsize,input_seq-raw_seq),dtype=np.int32)))
attention_mask=np.hstack((attention_mask,np.zeros((batchsize,input_seq-raw_seq),dtype=np.int32)))

print(tokens_id.shape,attention_mask.shape)
context.active_optimization_profile = 0
origin_inputshape = context.get_binding_shape(0)                # (1,-1) 
origin_inputshape[0],origin_inputshape[1] = tokens_id.shape     # (batch_size, max_sequence_length)
context.set_binding_shape(0, (origin_inputshape))               
context.set_binding_shape(1, (origin_inputshape))


inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
inputs[0].host = tokens_id
inputs[1].host = attention_mask

"""
d、tensorrt
"""
begin=datetime.now()
trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
print(trt_outputs)
end=datetime.now()
print("time:",(end-begin).total_seconds())
preds = np.argmax(trt_outputs, axis=1)
print("====preds====:",preds)