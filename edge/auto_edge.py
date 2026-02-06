# edge/auto_edge.py

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import psutil
import time
import onnxruntime as ort
import cv2
import numpy as np

MODEL_PATH = "models/mobilenetv3_sem.onnx"

CLASSES = [
    "clean","bridge","cmp","crack",
    "open","ler","via","other"
]

IMG_SIZE_HIGH = 224
IMG_SIZE_LOW  = 160

CPU_THRESHOLD = 65
OTHER_THRESHOLD = 0.75

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

def preprocess(img_path, size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0,1))
    return img

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def auto_edge_infer(img_path):
    cpu = psutil.cpu_percent(interval=0.1)

    size = IMG_SIZE_LOW if cpu > CPU_THRESHOLD else IMG_SIZE_HIGH
    mode = "LOW_RES" if cpu > CPU_THRESHOLD else "HIGH_RES"

    start = time.time()
    x = preprocess(img_path, size)
    logits = session.run(None, {input_name: x})[0][0]
    latency = time.time() - start

    probs = softmax(logits)
    cls_id = int(np.argmax(probs))
    confidence = float(probs[cls_id])
    raw = CLASSES[cls_id]

    pred = raw
    if confidence < OTHER_THRESHOLD:
        pred = "other"

    return {
        "class": pred,
        "raw": raw,
        "confidence": confidence,
        "cpu": cpu,
        "latency": latency,
        "mode": mode
    }

if __name__ == "__main__":
    img = "dataset/sample/test3.png"
    print(auto_edge_infer(img))
