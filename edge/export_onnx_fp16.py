# edge/export_onnx_fp16.py
import onnx
from onnxconverter_common import float16

FP32_ONNX = "models/convnext_sem.onnx"
FP16_ONNX = "models/convnext_sem_fp16.onnx"

print("🔧 Loading FP32 ONNX model...")
model = onnx.load(FP32_ONNX)

print(" Converting model to FP16...")
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True   # keep input/output as float32 for compatibility
)

onnx.save(model_fp16, FP16_ONNX)
print("✅ FP16 ONNX saved to:", FP16_ONNX)
