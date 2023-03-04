from pathlib import Path
import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# load model and tokenizer
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
feature = "sequence-classification"

model_id="textattack/bert-base-uncased-yelp-polarity"
save_name="bert-base"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path(save_name+".onnx")
)

print(onnx_inputs)
print(onnx_outputs)