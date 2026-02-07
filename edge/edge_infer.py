import sys, os
import onnxruntime as ort
import numpy as np
import time
from PIL import Image

# Setup Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

MODEL_PATH = "models/mobilenetv3_sem.onnx"

# FIXED: Strictly Alphabetical to match Training (ImageFolder)
CLASSES = [
    "bridge", "clean", "cmp", "crack", 
    "ler", "open", "other", "via"
]

IMG_SIZE = 224
OTHER_THRESHOLD = 0.0  # Set to 0.0 to verify 98% accuracy first

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(img_path):
    # Use PIL to match Training Grayscale and Resize math
    img = Image.open(img_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    
    # Scale 0-1 and convert to float32
    img_data = np.array(img).astype(np.float32) / 255.0
    # Shape: (1, 1, 224, 224)
    img_data = np.expand_dims(img_data, axis=(0, 1))
    return img_data

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def infer(img_path):
    x = preprocess(img_path)
    
    start = time.time()
    logits = session.run(None, {input_name: x})[0][0]
    latency = time.time() - start

    probs = softmax(logits)
    cls_id = int(np.argmax(probs))
    confidence = float(probs[cls_id])
    
    raw_class = CLASSES[cls_id]
    
    # Apply Threshold logic
    final_pred = raw_class
    if confidence < OTHER_THRESHOLD:
        final_pred = "other"

    return {
        "class": final_pred,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency * 1000, 2)
    }

if __name__ == "__main__":
    test_img = "dataset/sample/test4.png"
    if os.path.exists(test_img):
        print(infer(test_img))
    else:
        print(f"File not found: {test_img}")
