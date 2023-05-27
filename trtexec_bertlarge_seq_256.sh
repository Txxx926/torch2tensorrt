trtexec --onnx=/usr/local/onnx_models/bert_large.onnx \
        --saveEngine=/usr/local/onnx_models/bert_large_free_seq_bs_64.plan \
        --minShapes=input_ids:1x1,attention_mask:64x1,token_type_ids:64x1 \
        --optShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
        --maxShapes=input_ids:64x512,attention_mask:64x512,token_type_ids:64x512 \