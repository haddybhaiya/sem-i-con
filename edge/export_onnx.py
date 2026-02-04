import torch
import sys, os

# ------------------ PATH SETUP ------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from training.model import build_model   # must match training

# ------------------ CONFIG ------------------
MODEL_PATH = "models/convnext_sem.pth"      # trained ConvNeXt
ONNX_PATH  = "models/convnext_sem.onnx"

NUM_CLASSES = 8
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD MODEL ------------------
model = build_model(
    num_classes=NUM_CLASSES,
    in_chans=1           # IMPORTANT: grayscale
)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ------------------ DUMMY INPUT ------------------
dummy_input = torch.randn(
    1,        # batch
    1,        # channels (GRAY)
    IMG_SIZE,
    IMG_SIZE
)

# ------------------ EXPORT ------------------
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch"},
        "logits": {0: "batch"}
    }
)

print("✅ FP32 ONNX model exported to:", ONNX_PATH)
