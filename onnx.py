

import torch
from pathlib import Path
from transformers import BertTokenizer,BertModel
from transformers.convert_graph_to_onnx import convert
import os

def trans2onnx(origin_model_path):
  
    # Handles all the above steps for you
    convert(framework="pt",
            model=origin_model_path,
            output=Path("./onnx/bert-base-chinese.onnx"),
            opset=11)
  
    print("convert success")
    
def saveModel():
    origin_model_path = ""
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")

    output_dir = 'bert-base-chinese/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__=='__main__':
    origin_model_path = 'bert-base-chinese/'
    saveModel()
    trans2onnx(origin_model_path)