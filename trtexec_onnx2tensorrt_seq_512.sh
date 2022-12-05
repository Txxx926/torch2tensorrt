trtexec --onnx=distillbert.onnx \
        --saveEngine=distillbert_seq_512_min1x32_opt_8x128_max_64x512.plan \
        --minShapes=input_ids:1x512,attention_mask:1x32 \
        --optShapes=input_ids:8x512,attention_mask:8x128 \
        --maxShapes=input_ids:64x512,attention_mask:64x512 \