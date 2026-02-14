import os
import json
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# ⚙️ CONFIG & PATHS
# ===============================
MODEL_PATH = "models/mobilenetv3_sem.onnx"
DATASET_DIR = "dataset/hackathon_test_dataset"
IMG_SIZE = 224

CLASSES = ["bridge", "clean", "cmp", "crack", "ler", "open", "other", "via"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

FOLDER_TO_CLASS = {
    "Bridge": "bridge", "Clean": "clean", "CMP": "cmp", "Crack": "crack",
    "LER": "ler", "Open": "open", "Other": "other", "Particle": "other", "VIA": "via",
}

# --- HYPERPARAMETERS ---
T_SCALE = 0.65         # Sharpen peaks
OTHER_MARGIN = 0.12    # If 'other' is within 12% of the top class, pick 'other'
MIN_CONFIDENCE = 0.22  # Hard floor for any class; otherwise 'other'

# ===============================
# 🖼️ PREPROCESSING
# ===============================
def preprocess(img_path):
    try:
        img = Image.open(img_path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        # Scaled 0-1 to match your training ToTensor()
        img = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=(0, 1))
    except Exception as e:
        return None

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

# ===============================
# 🚀 INFERENCE LOOP
# ===============================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

y_true, y_pred = [], []
other_idx = CLASS_TO_ID["other"]

print(f"🚀 Starting Rank-Based Inference (T={T_SCALE}, Margin={OTHER_MARGIN})...")

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path): continue

    mapped_class = FOLDER_TO_CLASS.get(folder)
    if not mapped_class: continue
    true_id = CLASS_TO_ID[mapped_class]

    for img_name in tqdm(os.listdir(folder_path), desc=f"{folder}"):
        img_path = os.path.join(folder_path, img_name)
        x = preprocess(img_path)
        if x is None: continue

        # 1. Run Model
        logits = session.run(None, {input_name: x})[0][0]
        
        # 2. Apply Temperature
        probs = softmax(logits / T_SCALE)
        
        # 3. Get Rankings
        best_id = int(np.argmax(probs))
        other_prob = probs[other_idx]
        best_prob = probs[best_id]

        # --- RANK-BASED RESCUE LOGIC ---
        # If 'other' is not #1, but is very close to #1, reclassify as 'other'
        if best_id != other_idx and (best_prob - other_prob) < OTHER_MARGIN:
            pred_id = other_idx
        else:
            pred_id = best_id

        # --- FINAL SEMANTIC GUARD ---
        if probs[pred_id] < MIN_CONFIDENCE:
            pred_id = other_idx

        y_true.append(true_id)
        y_pred.append(pred_id)

# ===============================
# 📊 FINAL METRICS
# ===============================
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
