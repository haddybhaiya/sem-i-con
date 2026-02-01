import torch
import sys, os

# add project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from training.model import build_model

MODEL_PATH = "models/1edge_model.pth" #change in pipeline from torch-quantize.py
OUT_PATH = "models/1edge_model_int8.pth" #no longer going extra dynamic quantization step
NUM_CLASSES = 8

device = "cpu"

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Dynamic quantization (safe for CNN inference)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # quantize FC layers
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), OUT_PATH)
print("PyTorch INT8 model saved to:", OUT_PATH)
