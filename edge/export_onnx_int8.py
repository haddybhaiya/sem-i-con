import torch
import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from training.model import build_model

INT8_PTH = "models/sem.onnx"
INT8_ONNX = "models/sem_int8.onnx"

NUM_CLASSES = 8
IMG_SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(INT8_PTH, map_location=device))
model.eval()

dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    INT8_ONNX,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True
)

print("✅ INT8 ONNX exported to:", INT8_ONNX)
