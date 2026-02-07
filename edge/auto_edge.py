import sys, os
import psutil
import time
import onnxruntime as ort
import numpy as np
from PIL import Image

# Setup Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

MODEL_PATH = "models/mobilenetv3_sem.onnx"

# FIXED: Strictly Alphabetical
CLASSES = [
    "bridge", "clean", "cmp", "crack", 
    "ler", "open", "other", "via"
]

IMG_SIZE = 224 # Strictly use 224 to maintain accuracy
OTHER_THRESHOLD = 0.5 # Lowered to be less aggressive

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def auto_edge_infer(img_path):
    # Monitor CPU but keep resolution constant for accuracy
    cpu_usage = psutil.cpu_percent(interval=0.05)
    
    start = time.time()
    
    # Preprocess (PIL Logic)
    img = Image.open(img_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.array(img).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, 1))

    # Inference
    logits = session.run(None, {input_name: x})[0][0]
    latency = time.time() - start

    # Post-process
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    
    cls_id = int(np.argmax(probs))
    confidence = float(probs[cls_id])
    
    raw_class = CLASSES[cls_id]
    pred = raw_class if confidence >= OTHER_THRESHOLD else "other"

    return {
        "class": pred,
        "raw": raw_class,
        "confidence": round(confidence, 4),
        "cpu_usage": cpu_usage,
        "latency_ms": round(latency * 1000, 2)
    }

if __name__ == "__main__":
    test_img = "dataset/sample/test3.png"
    if os.path.exists(test_img):
        print(auto_edge_infer(test_img))
