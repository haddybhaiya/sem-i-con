import os
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# --- CONFIG ---
MODEL_PATH = "models/mobilenetv3_sem_distilled.pth"
TEST_DIR = "dataset/test"
NUM_CLASSES = 8
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. MATCH TRAINING TRANSFORMS EXACTLY
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# 2. LOAD DATASET (Alphabetical by default)
test_ds = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
print(f"Classes found: {test_ds.classes}")

# 3. RECONSTRUCT MODEL SKELETON
model = timm.create_model(
    "mobilenetv3_small_100",
    pretrained=False, # We load our weights anyway
    in_chans=1,
    num_classes=NUM_CLASSES
)

# 4. LOAD WEIGHTS
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval() # CRITICAL: Sets Batch Norm layers to inference mode

# 5. EVALUATION LOOP
correct = 0
total = 0

print(f"🚀 Evaluating .pth model on {DEVICE}...")
with torch.no_grad():
    for x, y in tqdm(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

final_accuracy = correct / total
print("\n" + "="*30)
print(f"✅ PyTorch (.pth) Accuracy: {final_accuracy:.4f}")
print("="*30)
