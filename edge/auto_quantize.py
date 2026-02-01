from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_PATH = "models/1edge_model.onnx" #newly exported ONNX model
OUT_PATH = "models/1edge_model_int8.onnx"

quantize_dynamic(
    model_input=MODEL_PATH,
    model_output=OUT_PATH,
    weight_type=QuantType.QInt8,
    extra_options={"DisableShapeInference": True}  # Optional: speeds up quantization(first time only)
)

print("Quantized model saved to:", OUT_PATH)
