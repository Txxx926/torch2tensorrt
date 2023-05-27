trtexec --onnx=bert_base.onnx \
        --saveEngine=bert_base_seq_192.plan \
        --minShapes=input_ids:1x192,attention_mask:1x192,token_type_ids:1x192 \
        --optShapes=input_ids:8x192,attention_mask:8x192,token_type_ids:8x192 \
        --maxShapes=input_ids:64x192,attention_mask:64x192,token_type_ids:64x192 \