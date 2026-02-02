import torch
import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from training.model import build_model

MODEL_PATH = "models/1edge_model.pth" #change in pipeline from torch-quantize.py
ONNX_PATH = "models/1edge_model_int8.onnx"
NUM_CLASSES = 8
IMG_SIZE = 224

device = "cpu"

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True
)

print("INT8 ONNX model exported to:", ONNX_PATH)
