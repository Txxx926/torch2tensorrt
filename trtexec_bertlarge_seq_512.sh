trtexec --onnx=/usr/local/onnx_models/bert_large.onnx \
        --saveEngine=/usr/local/onnx_models/bert_large_seqfree_bs_1.plan \
        --minShapes=input_ids:32x512,attention_mask:32x512,token_type_ids:32x512 \
        --optShapes=input_ids:32x512,attention_mask:32x512,token_type_ids:32x512 \
        --maxShapes=input_ids:32x512,attention_mask:32x512,token_type_ids:32x512 \