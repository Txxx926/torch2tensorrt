trtexec --onnx=distillbert.onnx \
        --saveEngine=distillbert_seq_128.plan \
        --minShapes=input_ids:1x128,attention_mask:1x128 \
        --optShapes=input_ids:8x128,attention_mask:8x128 \
        --maxShapes=input_ids:64x128,attention_mask:64x128 \