trtexec --onnx=distillbert.onnx \
        --saveEngine=distillbert_seq_320.plan \
        --minShapes=input_ids:1x320,attention_mask:1x320 \
        --optShapes=input_ids:8x320,attention_mask:8x320 \
        --maxShapes=input_ids:64x320,attention_mask:64x320 \