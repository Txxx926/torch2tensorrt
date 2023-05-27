trtexec --onnx=bert_large.onnx \
        --saveEngine=bert_large_seq64.plan \
        --minShapes=input_ids:1x64,attention_mask:1x64,token_type_ids:1x64 \
        --optShapes=input_ids:8x64,attention_mask:8x64,token_type_ids:8x64 \
        --maxShapes=input_ids:64x64,attention_mask:64x64,token_type_ids:64x64 \