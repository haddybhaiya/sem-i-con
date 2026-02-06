# edge/export_onnx.py

import torch
import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from training.model import build_model

MODEL_PATH = "models/mobilenetv3_sem_distilled.pth"
ONNX_PATH  = "models/mobilenetv3_sem.onnx"

NUM_CLASSES = 8
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(NUM_CLASSES)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=17
)

print("✅ Exported:", ONNX_PATH)
