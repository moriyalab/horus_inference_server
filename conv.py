import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30

        # FP16モードを使用する場合は以下を有効に
        # if builder.platform_has_fast_fp16:
        #     config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        engine = builder.build_engine(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

onnx_file_path = "./ml_model/coco.onnx"  # ONNXファイルのパス
engine_file_path = "./ml_model/coco2.engine"  # 出力エンジンファイルのパス

engine = build_engine(onnx_file_path, engine_file_path)
if engine:
    print("Engine has been successfully built and saved.")
else:
    print("Failed to build the engine.")