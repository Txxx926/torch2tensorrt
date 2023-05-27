trtexec --onnx=/usr/local/onnx_models/bert_base.onnx \
        --saveEngine=/usr/local/onnx_models/bert_base_seq_384_fixed_fp16.plan \
        --fp16 \
        --minShapes=input_ids:32x384,attention_mask:32x384,token_type_ids:32x384 \
        --optShapes=input_ids:32x384,attention_mask:32x384,token_type_ids:32x384 \
        --maxShapes=input_ids:32x384,attention_mask:32x384,token_type_ids:32x384 \