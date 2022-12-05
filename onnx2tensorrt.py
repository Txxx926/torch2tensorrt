import tensorrt as trt
def convert_onnx_to_engine(onnx_filename,
                           engine_filename = None,
                           max_batch_size = 32,
                           max_workspace_size = 1 << 30,
                           fp16_mode = False):
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, \
            builder.create_network() as network, \
            trt.OnnxParser(network, logger) as parser:

        builder.max_workspace_size = max_workspace_size
        builder.fp16_mode = fp16_mode
        builder.max_batch_size = max_batch_size

        print("Parsing ONNX file.")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print('Completed parsing of ONNX file')

        print("Building TensorRT engine. This may take a few minutes.")
        engine = builder.build_cuda_engine(network)
        print("Created engine success! ")

        if engine_filename:
            with open(engine_filename, 'wb') as f:
                f.write(engine.serialize())

        return engine, logger


convert_onnx_to_engine("distillbert.onnx","distillbert.plan",max_batch_size=128)