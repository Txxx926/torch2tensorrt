trtexec --onnx=/usr/local/onnx_models/bert_large.onnx \
        --saveEngine=/usr/local/onnx_models/bert_large_seq128_bs_64.plan \
        --minShapes=input_ids:64x128,attention_mask:64x128,token_type_ids:64x128 \
        --optShapes=input_ids:64x128,attention_mask:64x128,token_type_ids:64x128 \
        --maxShapes=input_ids:64x128,attention_mask:64x128,token_type_ids:64x128 \