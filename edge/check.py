import torch
import numpy as np
import onnxruntime as ort
import os, sys

# Ensure your model architecture is accessible
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from training.model import build_model # Adjust this to your MobileNetV3 build function

# 1. Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/mobilenetv3_sem_distilled.pth"
ONNX_PATH = "models/mobilenetv3_sem.onnx"

# 2. Load PyTorch Model
model = build_model(num_classes=8) # Use your student build function here
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 3. Create Dummy Input (Match your export: 1 channel, 224x224)
dummy = torch.randn(1, 1, 224, 224).to(DEVICE)

# 4. Get PyTorch output
with torch.no_grad():
    torch_out = model(dummy)

# 5. Get ONNX output
# Use 'CPUExecutionProvider' if you don't have onnxruntime-gpu installed
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
onnx_inputs = {session.get_inputs()[0].name: dummy.cpu().numpy()}
onnx_out = session.run(None, onnx_inputs)[0]

# 6. Compare
try:
    np.testing.assert_allclose(torch_out.cpu().numpy(), onnx_out, rtol=1e-03, atol=1e-05)
    print("✅ ONNX Match Success! The export is perfect.")
except AssertionError as e:
    print("❌ ONNX Match Failed! The export is corrupted or mismatched.")
    print(e)
