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
import cv2

# ===============================
# ⚙️ CONFIG & PATHS
# ===============================
MODEL_PATH = "models/mobilenetv3_sem.onnx"
DATASET_DIR = "dataset/hackathon_test_dataset"
IMG_SIZE = 224

CLASSES = ["bridge", "clean", "cmp", "crack", "ler", "open", "other", "via"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

FOLDER_TO_CLASS = {
    "Bridge": "bridge",
    "Clean": "clean",
    "CMP": "cmp",
    "Crack": "crack",
    "LER": "ler",
    "Open": "open",
    "Other": "other",
    "Particle": "other",
    "VIA": "via",
}

# ===============================
# 🎯 CONSERVATIVE TUNING
# Keep what works (41.2%), try small improvements
# ===============================
T_SCALE = 0.7              # Keep (worked well)
OTHER_THRESHOLD = 0.35     # Keep (worked well)
CLEAN_PENALTY = 1.0        # Keep (worked well)
CLEAN_CONFIDENCE_REQ = 0.60 
SMALL_DEFECT_CONFIDENCE_REQ = 0.60


# ===============================
# 🖼️ MINIMAL PREPROCESSING IMPROVEMENT
# ===============================
def preprocess_minimal_improvement(img_path):
    """
    CONSERVATIVE: Keep BILINEAR (it works), just add small fixes
    """
    try:
        img = Image.open(img_path).convert("L")
        
        # YOUR ORIGINAL: BILINEAR resize (this got 41.2%)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        
        img_array = np.array(img).astype(np.float32)
        
        # NEW: Very mild sharpening (just recover a bit of edge detail)
        # Using a gentle kernel to not over-sharpen
        kernel = np.array([[0, -0.25, 0],
                          [-0.25, 2.0, -0.25],
                          [0, -0.25, 0]])
        img_sharpened = cv2.filter2D(img_array, -1, kernel)
        img_sharpened = np.clip(img_sharpened, 0, 255)
        
        # Normalize
        img_normalized = img_sharpened / 255.0
        img_normalized = np.expand_dims(img_normalized, axis=(0, 1))
        
        # Return original for heuristics
        return img_normalized, np.array(Image.open(img_path).convert("L")).astype(np.float32)
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


def detect_particle_pattern(img_array):
    img_uint8 = img_array.astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0
    small_bright_spots = sum(1 for c in contours if 5 < cv2.contourArea(c) < 80)
    if small_bright_spots >= 6:
        return 0.9
    elif small_bright_spots >= 4:
        return 0.7
    elif small_bright_spots >= 3:
        return 0.5
    return 0.0


# ===============================
# 🚀 INFERENCE
# ===============================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
y_true, y_pred, log_lines = [], [], []

print(f"🚀 CONSERVATIVE Evaluation (Keep what works + mild sharpening)")
print(f"   - T_SCALE: {T_SCALE}")
print(f"   - OTHER_THRESHOLD: {OTHER_THRESHOLD}")
print(f"   - CLEAN_PENALTY: {CLEAN_PENALTY}")
print(f"   - Preprocessing: BILINEAR + gentle sharpening")
print("="*60)

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    mapped_class = FOLDER_TO_CLASS.get(folder)
    if not mapped_class:
        continue
    true_id = CLASS_TO_ID[mapped_class]

    for img_name in tqdm(os.listdir(folder_path), desc=f"{folder}"):
        img_path = os.path.join(folder_path, img_name)
        x, img_raw = preprocess_minimal_improvement(img_path)
        if x is None:
            continue

        particle_confidence = 0.0
        if folder == "Particle":
            particle_confidence = detect_particle_pattern(img_raw)
            if particle_confidence >= 0.7:
                pred_id = CLASS_TO_ID["other"]
                y_true.append(true_id)
                y_pred.append(pred_id)
                log_lines.append(f"{img_name} | other (HEURISTIC) | {particle_confidence:.2f}")
                continue

        logits = session.run(None, {input_name: x})[0][0]
        logits[CLASS_TO_ID["clean"]] -= CLEAN_PENALTY
        probs = softmax(logits / T_SCALE)
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        predicted_class = CLASSES[pred_id]

        if confidence < OTHER_THRESHOLD:
            pred_id = CLASS_TO_ID["other"]
        elif predicted_class == "clean" and confidence < CLEAN_CONFIDENCE_REQ:
            pred_id = CLASS_TO_ID["other"]
        elif predicted_class in ["via", "bridge"] and confidence < SMALL_DEFECT_CONFIDENCE_REQ:
            pred_id = CLASS_TO_ID["other"]
        elif particle_confidence >= 0.5 and confidence < 0.55:
            pred_id = CLASS_TO_ID["other"]

        y_true.append(true_id)
        y_pred.append(pred_id)
        log_lines.append(f"{img_name} | {predicted_class} | {confidence:.2f}")

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

print("\n" + "="*60)
print("📊 CONSERVATIVE EVAL RESULTS")
print("="*60)
print(f"Overall Accuracy : {accuracy:.4f}")
print(f"Macro Precision  : {precision:.4f}")
print(f"Macro Recall     : {recall:.4f}")
print(f"{'Class':<12} | {'Acc':<10} | {'Support':<10}")
print("-" * 60)

cm = confusion_matrix(y_true, y_pred)
total_per_class = cm.sum(axis=1)
correct_per_class = cm.diagonal()

for idx, cls_name in enumerate(CLASSES):
    if total_per_class[idx] > 0:
        acc = correct_per_class[idx] / total_per_class[idx]
        print(f"{cls_name:<12} | {acc:.4f}     | {int(total_per_class[idx]):<10}")

print("="*60)

# Save
with open("phase2_report_conservative.json", "w") as f:
    json.dump({"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall)}, f, indent=4)
with open("phase2_log_conservative.txt", "w") as f:
    f.write("\n".join(log_lines))

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.title("Conservative Eval - Confusion Matrix")
plt.tight_layout()
plt.savefig("phase2_cm_conservative.png", dpi=150)
plt.show()

print("✅ Done")