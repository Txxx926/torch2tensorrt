trtexec --onnx=bert_base.onnx \
        --saveEngine=bert_base_seq_256.plan \
        --minShapes=input_ids:1x256,attention_mask:1x256,token_type_ids:1x256 \
        --optShapes=input_ids:8x256,attention_mask:8x256,token_type_ids:8x256 \
        --maxShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \