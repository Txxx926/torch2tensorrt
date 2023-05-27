from pathlib import Path
import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
# load model and tokenizer
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
feature = "sequence-classification"

model_id="textattack/bert-base-uncased-yelp-polarity"

model_id="bert-large-uncased"
save_name="bert_large"
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

low_level=True
#dummy_model_input = tokenizer("This is a sample", return_tensors="pt")
#print(dummy_model_input)

dummy_model_input = {'input_ids':torch.randint(10,1000,(32,512)).long(),  'attention_mask':torch.randint(1,2,(32,512)).long() }#,token_type_ids':torch.randint(1,2,(32,512)).long() }
for x,v in dummy_model_input.items():
        print(v.shape)
if low_level:
        #dummy_model_input = tokenizer("This is a sample", return_tensors="pt")
        torch.onnx.export(
                model, 
                tuple(dummy_model_input.values()),
                f="/usr/local/onnx_models/distillbert_seq_512.onnx",  
                input_names=['input_ids', 'attention_mask'],#,'token_type_ids'], 
                output_names=['logits'], 
                dynamic_axes={'input_ids': {0: 'batch_size'}, 
                  'attention_mask': {0: 'batch_size'}, 
                 # 'token_type_ids': {0: 'batch_size'},
                  'logits': {0: 'batch_size'}}, 
                do_constant_folding=True, 
                opset_version=13, 
        )
# export
else:
        onnx_inputs, onnx_outputs = transformers.onnx.export(
               preprocessor=tokenizer,
               model=model,
           config=onnx_config,
              opset=13,
            output=Path(save_name+".onnx")
        )

#print(onnx_inputs)
#print(onnx_outputs)