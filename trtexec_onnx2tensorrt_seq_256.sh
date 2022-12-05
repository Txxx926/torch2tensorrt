trtexec --onnx=distillbert.onnx \
        --saveEngine=distillbert_seq_256.plan \
        --minShapes=input_ids:1x256,attention_mask:1x256 \
        --optShapes=input_ids:8x256,attention_mask:8x256 \
        --maxShapes=input_ids:64x256,attention_mask:64x256 \