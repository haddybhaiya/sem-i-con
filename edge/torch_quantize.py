import torch
import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from training.model import build_model

FP32_PTH = "models/sem.pth"
INT8_PTH = "models/sem_int8.pth"
NUM_CLASSES = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(FP32_PTH, map_location=device))
model.eval()

# ✅ Quantize ONLY Linear layers (safe for EfficientNet)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), INT8_PTH)

print("✅ PyTorch INT8 model saved to:", INT8_PTH)
