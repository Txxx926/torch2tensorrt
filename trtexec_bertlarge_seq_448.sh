trtexec --onnx=bert_large.onnx \
        --saveEngine=bert_large_seq448.plan \
        --minShapes=input_ids:1x448,attention_mask:1x448,token_type_ids:1x448 \
        --optShapes=input_ids:8x448,attention_mask:8x448,token_type_ids:8x448 \
        --maxShapes=input_ids:64x448,attention_mask:64x448,token_type_ids:64x448 \