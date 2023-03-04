import numpy as np
import onnx
import torch
import onnxruntime as rt
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime


#create runtime session
sess = rt.InferenceSession("distillbert.onnx",providers=['CUDAExecutionProvider'])

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

model_id="textattack/bert-base-uncased-yelp-polarity"

model = AutoModelForSequenceClassification.from_pretrained(model_id).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)

input=tokenizer.batch_encode_plus(["i hate Shake. He is a bad guy in the baskerball game","i hate Shake. He is a bad guy in the baskerball game"])
token_id=torch.Tensor(input['input_ids']).long().cuda()
mask=torch.Tensor(input['attention_mask']).long().cuda()
print(input)
begin=datetime.now()
output=model(token_id,mask)
torch.cuda.synchronize()
end=datetime.now()
print(output,"time:",(end-begin).total_seconds())
# get output name
input_name0 = sess.get_inputs()[0].name
input_shape0 = sess.get_inputs()[0].shape
print("input name0", input_name0,input_shape0)
input_name1 = sess.get_inputs()[1].name
print("input name1", input_name1)
output_name= sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)

#forward model

input_data0=token_id.cpu().numpy()
input_data1=mask.cpu().numpy()
#input_data0=np.random.randint(1000,size=(1,128)).astype("int64")
#input_data1=np.ones((1,128)).astype("int64")
print(rt.get_device())
begin=datetime.now()
res = sess.run([output_name], {input_name0: input_data0,input_name1:input_data1})
end=datetime.now()
print("time:",(end-begin).total_seconds())
out = np.array(res)
print(out)