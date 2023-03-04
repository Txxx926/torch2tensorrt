trtexec --onnx=distillbert.onnx \
        --saveEngine=distillbert_seq_384.plan \
        --minShapes=input_ids:1x384,attention_mask:1x384 \
        --optShapes=input_ids:8x384,attention_mask:8x384 \
        --maxShapes=input_ids:64x384,attention_mask:64x384 \