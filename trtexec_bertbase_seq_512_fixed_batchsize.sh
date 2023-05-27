trtexec --onnx=/usr/local/onnx_models/bert_large.onnx \
        --saveEngine=bert_large_seq_256_fixed_batchsize.plan \
        --minShapes=input_ids:64x64,attention_mask:64x64,token_type_ids:64x64 \
        --optShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
        --maxShapes=input_ids:64x512,attention_mask:64x512,token_type_ids:64x512 \