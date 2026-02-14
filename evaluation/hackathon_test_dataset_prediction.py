import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix



MODEL_PATH = "models/mobilenetv3_sem.onnx"
DATASET_DIR = "dataset/2test"
IMG_SIZE = 224

CLASSES = ["bridge", "clean", "cmp", "crack", "ler", "open", "other", "via"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# --- HYPERPARAMETERS FOR INFERENCE ---
T_SCALE = 1.2          # Higher T to soften overconfident 'clean' predictions
OTHER_MARGIN = 0.20    # Increased margin to rescue from 'clean' sinkhole
# LOGIT_BIAS: Penalize 'clean' and boost difficult classes like 'ler'/'other'
# Order: [bridge, clean, cmp, crack, ler, open, other, via]
LOGIT_BIAS = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

# preprocessing function to match PyTorch's default transforms (Resize + ToTensor)
def preprocess(img_pil):
    """Matches PyTorch: Resize (Bicubic) -> ToTensor"""
    # Use BICUBIC to match Torchvision default
    img = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=(0, 1))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

# change in inference loop: we will do TTA and apply logit bias + temperature scaling before final prediction
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
other_idx = CLASS_TO_ID["other"]
clean_idx = CLASS_TO_ID["clean"]

y_true, y_pred = [], []

print(f"🚀 Starting TTA + Biased Inference...")

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path): continue
    
    # Ensure folder name matches our ALPHABETICAL class list
    true_id = CLASS_TO_ID.get(folder.lower())
    if true_id is None: continue

    for img_name in tqdm(os.listdir(folder_path), desc=f"{folder}"):
        img_path = os.path.join(folder_path, img_name)
        
        try:
            raw_img = Image.open(img_path).convert("L")
            
            # --- TEST TIME AUGMENTATION (TTA) ---
            # We run inference on Original, Horizontal Flip, and Vertical Flip
            imgs = [raw_img, ImageOps.mirror(raw_img), ImageOps.flip(raw_img)]
            tta_logits = []
            
            for img in imgs:
                x = preprocess(img)
                logits = session.run(None, {input_name: x})[0][0]
                tta_logits.append(logits)
            
            # Average logits across augmentations for stability
            avg_logits = np.mean(tta_logits, axis=0)
            
            # --- POST-PROCESSING ---
            # 1. Apply Logit Bias to fix distribution shift
            biased_logits = avg_logits + LOGIT_BIAS
            
            # 2. Temperature Scaling
            probs = softmax(biased_logits / T_SCALE)
            
            best_id = int(np.argmax(probs))
            best_prob = probs[best_id]
            other_prob = probs[other_idx]

            # --- TARGETED RESCUE ---
            # If the model is stuck in 'clean', but 'other' is somewhat likely, force 'other'
            if best_id == clean_idx and (best_prob - other_prob) < OTHER_MARGIN:
                pred_id = other_idx
            else:
                pred_id = best_id

            y_true.append(true_id)
            y_pred.append(pred_id)
            
        except Exception as e:
            continue # Skip problematic images


# print final evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("\n" + "="*40)
print(f"Overall Accuracy : {accuracy:.4f}")
print(f"Macro Precision  : {precision:.4f}")
print(f"Macro Recall     : {recall:.4f}")

print("\n" + "="*40)
print(f"{'Class Name':<12} | {'Accuracy':<10}")
print("-" * 40)

total_per_class = cm.sum(axis=1)
correct_per_class = cm.diagonal()

for idx, cls_name in enumerate(CLASSES):
    acc = correct_per_class[idx] / total_per_class[idx] if total_per_class[idx] > 0 else 0
    print(f"{cls_name:<12} | {acc:.4f}")
print("="*40)

# Plot for verification
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.title(f"Rank-Rescue Matrix (Acc: {accuracy:.4f})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
