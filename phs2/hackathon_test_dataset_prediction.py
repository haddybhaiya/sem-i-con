import os
import json
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_PATH = "models/mobilenetv3_sem.onnx"
DATASET_DIR = "dataset/hackathon_test_dataset"
IMG_SIZE = 224 #model trained on 224x224, but original images are 128x128, so we will resize during preprocessing
#note:  resizing always creates blurriness and loss of detail hence accuracy loss

# Optimization Hyperparameters
T_SCALE = 0.58  # Temperature Scaling (>1 softens distribution)
OTHER_THRESHOLD = 0.43  # If max confidence < this, classify as 'other' as discussed in the session

CLASSES = ["bridge", "clean", "cmp", "crack", "ler", "open", "other", "via"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
OTHER_ID = CLASS_TO_ID["other"]

FOLDER_TO_CLASS = {
    "Bridge": "bridge",
    "Clean": "clean",
    "CMP": "cmp",
    "Crack": "crack",
    "LER": "ler",
    "Open": "open",
    "Other": "other",
    "Particle": "other", # mapping particle to other 
    "VIA": "via",   
     
}



#preprocessing with PIL for optimized inference speed 
def preprocess_pil(img_path):
    """Resized 128px to 224px using PIL to match training pipeline."""
    # model accuracy loss due to resizing, but this is a requirement for the ONNX model input.
    with Image.open(img_path) as img:
        img = img.convert("L")
        # Use BILINEAR to match original training code
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_arr = np.array(img).astype(np.float32) / 255.0
        return img_arr[np.newaxis, np.newaxis, :, :]

def softmax_scaled(logits, temperature= 1.00):
    """Softmax with Temperature Scaling for calibration."""
    logits = np.array(logits) / temperature
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)

# Inference

# Load ONNX with optimized CPU settings
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

y_true, y_pred, log_lines = [], [], []

print(f"Running PIL-Optimized Inference (T={T_SCALE}, Thresh={OTHER_THRESHOLD})")

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path) or folder not in FOLDER_TO_CLASS:
        continue

    true_class = FOLDER_TO_CLASS[folder]
    true_id = CLASS_TO_ID[true_class]

    for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        img_path = os.path.join(folder_path, img_name)
        
        # Preprocess
        x = preprocess_pil(img_path)
        
        # Run Inference
        logits = session.run(None, {input_name: x})[0][0]
        
        # Apply Scaling & Threshold
        probs = softmax_scaled(logits, T_SCALE)
        max_prob = np.max(probs)
        raw_pred_id = np.argmax(probs)

        # Optimization: Classify as 'other' if confidence is too low
        if max_prob < OTHER_THRESHOLD:
            pred_id = OTHER_ID
        else:
            pred_id = int(raw_pred_id)

        y_true.append(true_id)
        y_pred.append(pred_id)
        
        log_lines.append(
            f"{img_name} | True: {true_class} | Pred: {CLASSES[pred_id]} | Conf: {max_prob:.4f}"
        )

# metrics and logging

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"\n📊 Results:\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print("class wise accuracy with support for each class:")

# Calculate per-class accuracy from the confusion matrix
total_per_class = cm.sum(axis=1)
correct_per_class = cm.diagonal()

for idx, cls_name in enumerate(CLASSES):
    if total_per_class[idx] > 0:
        acc = correct_per_class[idx] / total_per_class[idx]
        print(f"{cls_name}: {acc:.4f} (Support: {total_per_class[idx]})")
    else:
        print(f"{cls_name}: No Samples")

# Save Log and JSON Report
with open("prediction_log.txt", "w") as f:
    f.write("\n".join(log_lines))

with open("phase2_report.json", "w") as f:
    json.dump({"accuracy": accuracy, "precision": precision, "recall": recall}, f, indent=4)

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.title(f"Confusion Matrix (Threshold: {OTHER_THRESHOLD})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("phase2_confusion_matrix.png")
plt.show()

