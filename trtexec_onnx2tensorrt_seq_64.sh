trtexec --onnx=distillbert.onnx \
        --saveEngine=distillbert_seq_64.plan \
        --minShapes=input_ids:1x64,attention_mask:1x64 \
        --optShapes=input_ids:8x64,attention_mask:8x64 \
        --maxShapes=input_ids:64x64,attention_mask:64x64 \