import torch
import timm

MODEL_PATH = "models/sem.pth"
ONNX_PATH  = "models/sem.onnx"

NUM_CLASSES = 8
IMG_SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ MUST MATCH TRAINING EXACTLY
model = timm.create_model(
    "tf_efficientnetv2_s",   # <-- same backbone as training
    pretrained=True,
    num_classes=NUM_CLASSES,
    in_chans=3 
             # grayscale replicated → RGB
)

state = torch.load(MODEL_PATH, map_location=device)

# If trained normally (model.state_dict())
model.load_state_dict(state)

model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True
)

print("✅ Exported:", ONNX_PATH)
