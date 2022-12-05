trtexec --loadEngine=distillbert_seq_128.plan --shapes=input_ids:10x128,attention_mask:10x128
trtexec --loadEngine=distillbert_seq_256.plan --shapes=input_ids:10x256,attention_mask:10x256