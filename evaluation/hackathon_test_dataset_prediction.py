import os
import sys
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

# Model Classes (Must match alphabetical ImageFolder order from training)
CLASSES = ["bridge", "clean", "cmp", "crack", "ler", "open", "other", "via"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Hackathon Folder Mapping
FOLDER_TO_CLASS = {
    "Bridge": "bridge",
    "Clean": "clean",
    "CMP": "cmp",
    "Crack": "crack",
    "LER": "ler",
    "Open": "open",
    "Other": "other",    # Direct match
    "Particle": "other", # Mapped to existing 'other' class
    "VIA": "via",
}

# Hyperparameters for Inference
T_SCALE = 1.2       # Sharpen distilled model peaks
OTHER_THRESHOLD = 0.45  # Force to 'other' if max confidence is low


# PREPROCESSING

def preprocess(img_path):
    """Matches training: Grayscale -> Resize -> ToTensor (0-1 scale)"""
    try:
        img = Image.open(img_path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        # Scale to 0-1 (matches transforms.ToTensor() from your pipeline)
        img = np.array(img).astype(np.float32) / 255
        # Add Batch [N] and Channel [C] dims: [1, 1, 224, 224]
        return np.expand_dims(img, axis=(0, 1))
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

#inference
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

y_true, y_pred, log_lines = [], [], []

print(f"Starting Phase-2 Evaluation on: {DATASET_DIR}")

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path): continue

    mapped_class = FOLDER_TO_CLASS.get(folder)
    if not mapped_class:
        print(f"⚠️ Unknown folder '{folder}' - skipping.")
        continue

    true_id = CLASS_TO_ID[mapped_class]

    for img_name in tqdm(os.listdir(folder_path), desc=f"Evaluating {folder}"):
        img_path = os.path.join(folder_path, img_name)
        x = preprocess(img_path)
        if x is None: continue

        # Run ONNX
        logits = session.run(None, {input_name: x})[0][0]

        # If the model is guessing 'clean' but it's not a strong guess, 
        # we nudge it toward 'other'
        clean_idx = CLASS_TO_ID["clean"]
        logits[clean_idx] -= 0.5  # Artificial penalty to reduce 'clean' false positives
        
        # Apply Temperature and Softmax
        probs = softmax(logits / T_SCALE)
        
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

        # SEMANTIC GUARD: If it's a weak prediction, default to 'other' to avoid false positives
        if confidence < OTHER_THRESHOLD:
            pred_id = CLASS_TO_ID["other"]

        y_true.append(true_id)
        y_pred.append(pred_id)
        log_lines.append(f"{img_name} | True: {mapped_class} | Pred: {CLASSES[pred_id]} | Conf: {confidence:.2f}")


accuracy = accuracy_score(y_true, y_pred)
# Use zero_division=0 to handle classes not present in test set
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

# print(f"\nResults:\nAcc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

#  FINAL METRICS & CLASS-WISE BREAKDOWN

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

print("\n" + "="*40)
print("📊 PHASE-2 GLOBAL METRICS")
print("="*40)
print(f"Overall Accuracy : {accuracy:.4f}")
print(f"Macro Precision  : {precision:.4f}")
print(f"Macro Recall     : {recall:.4f}")

print("\n" + "="*40)
print(f"{'Class Name':<12} | {'Accuracy':<10}")
print("-" * 40)

# Calculate per-class accuracy from the confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Avoid division by zero if a class has 0 samples in the test set
total_per_class = cm.sum(axis=1)
correct_per_class = cm.diagonal()

for idx, cls_name in enumerate(CLASSES):
    if total_per_class[idx] > 0:
        acc = correct_per_class[idx] / total_per_class[idx]
        print(f"{cls_name:<12} | {acc:.4f}")
    else:
        print(f"{cls_name:<12} | No Samples")

print("="*40 + "\n")


# Save results
with open("phase2_report.json", "w") as f:
    json.dump({"accuracy": accuracy, "precision": precision, "recall": recall}, f, indent=4)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", 
            xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hackathon Evaluation Matrix")
plt.savefig("phase2_confusion_matrix.png")
plt.show()
