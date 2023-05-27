from transformers import BertTokenizer, BertModel
import time
import numpy as np
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to("cuda")
text = ["Replace me by any text you'd like."]*16
#print(model)
encoded_input = tokenizer(text, return_tensors='pt',padding='max_length',max_length=512)#.to("cuda")
model.eval()
print(encoded_input["input_ids"].shape)

#jit_model=torch.jit.script(model)
times=[]
for i in range(20):
    begin=time.time()
    encoded_input=encoded_input.to("cuda")
    output = model(**encoded_input)

    end=time.time()
    if i >5:
        times.append((end-begin)*1000)
print(np.mean(times))
