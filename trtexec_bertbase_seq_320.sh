trtexec --onnx=bert_base.onnx \
        --saveEngine=bert_base_seq_320.plan \
        --minShapes=input_ids:1x320,attention_mask:1x320,token_type_ids:1x320 \
        --optShapes=input_ids:8x320,attention_mask:8x320,token_type_ids:8x320 \
        --maxShapes=input_ids:64x320,attention_mask:64x320,token_type_ids:64x320 \