trtexec --onnx=bert_large.onnx \
        --saveEngine=bert_large_seq384.plan \
        --minShapes=input_ids:1x384,attention_mask:1x384,token_type_ids:1x384 \
        --optShapes=input_ids:8x384,attention_mask:8x384,token_type_ids:8x384 \
        --maxShapes=input_ids:64x384,attention_mask:64x384,token_type_ids:64x384 \