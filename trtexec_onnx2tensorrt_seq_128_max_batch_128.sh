trtexec --onnx=distillbert.onnx \
        --saveEngine=distillbert_seq_128_max_batch_128.plan \
        --minShapes=input_ids:1x128,attention_mask:1x128 \
        --optShapes=input_ids:32x128,attention_mask:32x128 \
        --maxShapes=input_ids:128x128,attention_mask:128x128 \