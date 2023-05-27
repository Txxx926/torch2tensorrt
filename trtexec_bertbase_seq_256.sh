trtexec --onnx=/usr/local/onnx_models/bert_base.onnx \
        --saveEngine=/usr/local/onnx_models/bert_base_seq_256_fp16.plan \
        -- fp16 \
        --minShapes=input_ids:1x256,attention_mask:1x256,token_type_ids:1x256 \
        --optShapes=input_ids:32x256,attention_mask:32x256,token_type_ids:32x256 \
        --maxShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \