trtexec --onnx=/usr/local/onnx_models/distillbert.onnx \
        --saveEngine=distillbert_seq_freesize.plan \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:32x256,attention_mask:32x256 \
        --maxShapes=input_ids:64x512,attention_mask:64x512 \