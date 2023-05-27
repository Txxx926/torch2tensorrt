import numpy as np
import onnx
import torch
import onnxruntime as rt
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime


#create runtime session
sess = rt.InferenceSession("/usr/local/onnx_models/distillbert_seq_512.onnx",providers=['CUDAExecutionProvider'])

#sess = rt.InferenceSession("/usr/local/onnx_models/bert_large_seq_len512.onnx",providers=['CUDAExecutionProvider'])


model_id = "distilbert-base-uncased-finetuned-sst-2-english"

model_id="textattack/bert-base-uncased-yelp-polarity"

"""
model_id="bert-large-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_id).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)

input=tokenizer.batch_encode_plus(["i hate Shake. He is a bad guy in the baskerball game","i hate Shake. He is a bad guy in the baskerball game"])
token_id=torch.Tensor(input['input_ids']).long().cuda()
mask=torch.Tensor(input['attention_mask']).long().cuda()
token_type_ids=torch.Tensor(input['token_type_ids']).long().cuda()
print(input)
begin=datetime.now()
output=model(token_id,mask,token_type_ids)
torch.cuda.synchronize()
end=datetime.now()
print(output,"time:",(end-begin).total_seconds())
# get output name
"""
input_name0 = sess.get_inputs()[0].name
input_shape0 = sess.get_inputs()[0].shape
print("input name0", input_name0,input_shape0)
input_name1 = sess.get_inputs()[1].name
print("input name1", input_name1)
#input_name2 = sess.get_inputs()[2].name
#print("input name2", input_name2)
output_name= sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)

#forward model
from collections import defaultdict
latencies=defaultdict(dict)
for batchsize in range(1,65):
    for seq_len in range(512,513):
        input_data0=np.random.rand(batchsize,seq_len).astype(np.int64)
        input_data1=np.ones_like(input_data0).astype(np.int64)
    #input_data2=np.ones_like(input_data0).astype(np.int64)
#input_data0=np.random.randint(1000,size=(1,128)).astype("int64")
#input_data1=np.ones((1,128)).astype("int64")
        #print(rt.get_device())
        #print(input_data0.shape)
        begin=datetime.now()
        res = sess.run([output_name], {input_name0: input_data0,input_name1:input_data1})#,input_name2:input_data2})
        end=datetime.now()
        #print("time:",(end-begin).total_seconds())
        latencies[batchsize][seq_len]=(end-begin).total_seconds()*1000
    print(latencies)
        #out = np.array(res)
    #print(out.shape)
    
import json
model_name="distillbert_seq_512_onnx"
    # Data to be written
dictionary = {
        "model": model_name,
        "batchsize": "1-64",
        "seq_len":"1-512",
        "latency":latencies
    }
 
json_object = json.dumps(dictionary, indent=4)
with open(f"{model_name}_all_seqlen.json", "w") as outfile:
        outfile.write(json_object)
        