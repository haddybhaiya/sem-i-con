# evaluation/phase1_eval.py
# always load via PIL to match torchvision's Grayscale math

import os
import sys
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from PIL import Image

# Setup Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from evaluation.metrics import compute_metrics

# --- CONFIG ---
MODEL_PATH = "models/mobilenetv3_sem.onnx"
DATASET_DIR = "dataset/2test"

# FIXED: Must be ALPHABETICAL to match ImageFolder logic
CLASSES = [
    "bridge", "clean", "cmp", "crack", 
    "ler", "open", "other", "via"
]

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224
OTHER_THRESHOLD = 0.0  # Set to 0.0 first to see raw model performance

# --- ONNX SESSION ---
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

def preprocess(img_path):
    """
    Matches torchvision: Grayscale -> Resize (Bilinear) -> ToTensor
    """
    try:
        # Load via PIL to match torchvision's Grayscale math
        img = Image.open(img_path).convert('L')
        # Match torchvision default Resize (Bilinear)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        # Scale 0-1 (Matches ToTensor)
        img = np.array(img).astype(np.float32) / 255.0
        # Add Batch and Channel dimensions [1, 1, 224, 224]
        img = np.expand_dims(img, axis=(0, 1))
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# --- EVALUATION LOOP ---
y_true, y_pred = [], []

print(f"🚀 Starting Evaluation on {DATASET_DIR}...")

for cls_name in CLASSES:
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        print(f"⚠️ Warning: Directory {cls_dir} not found. Skipping.")
        continue

    true_id = CLASS_TO_ID[cls_name]
    
    files = os.listdir(cls_dir)
    for img_name in tqdm(files, desc=f"Evaluating {cls_name}"):
        img_path = os.path.join(cls_dir, img_name)

        x = preprocess(img_path)
        if x is None:
            continue

        # Run ONNX Inference
        logits = session.run(None, {input_name: x})[0][0]

         # --- NEW TEMPERATURE SCALING LOGIC ---
        # T < 1 (e.g., 0.8) makes the model more "confident" in its top choice
        T = 0.8 
        probs = softmax(logits / T)
        # probs = softmax(logits)  # Original for comparison

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

        # Apply Semantic Guard
        if confidence < OTHER_THRESHOLD:
            pred_id = CLASS_TO_ID["other"]

        y_true.append(true_id)
        y_pred.append(pred_id)

# --- METRICS ---
if not y_true:
    print("❌ Error: No images were processed. Check your DATASET_DIR.")
else:
    accuracy, class_acc, report = compute_metrics(
        y_true, y_pred, CLASSES
    )

    print("\n" + "="*30)
    print("📊 Phase-1 Evaluation Results")
    print("="*30)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("-" * 30)
    print("Class-wise Accuracy:")
    for cls in CLASSES:
        print(f"{cls:>8} : {class_acc.get(cls, 0.0):.4f}")
    print("="*30)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Generates and saves a heatmap of the model's confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Create Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Semiconductor Defect Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    save_path = "evaluation_results.png"
    plt.savefig(save_path)
    print(f"📊 Confusion Matrix saved to: {save_path}")
    plt.show()
plot_confusion_matrix(y_true, y_pred, CLASSES)

