#trtexec --loadEngine=/usr/local/onnx_models/bert_base_seq_256_bs_free.plan --shapes=input_ids:32x64,attention_mask:32x64,token_type_ids:32x64 --verbose
trtexec --loadEngine=/usr/local/onnx_models/bert_base_seq_384_fp16.plan --shapes=input_ids:32x384,attention_mask:32x384,token_type_ids:32x384 --verbose
