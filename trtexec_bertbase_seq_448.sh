trtexec --onnx=bert_base.onnx \
        --saveEngine=bert_base_seq_448.plan \
        --minShapes=input_ids:1x448,attention_mask:1x448,token_type_ids:1x448 \
        --optShapes=input_ids:8x448,attention_mask:8x448,token_type_ids:8x448 \
        --maxShapes=input_ids:64x448,attention_mask:64x448,token_type_ids:64x448 \