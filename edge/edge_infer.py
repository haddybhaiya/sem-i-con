# edge/edge_infer.py

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import onnxruntime as ort
import cv2
import numpy as np
import time

MODEL_PATH = "models/mobilenetv3_sem.onnx"

CLASSES = [
    "clean","bridge","cmp","crack",
    "open","ler","via","other"
]

IMG_SIZE = 224
OTHER_THRESHOLD = 0.75

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0,1))  # (1,1,H,W)
    return img

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def infer(img_path):
    x = preprocess(img_path)

    start = time.time()
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
        "latency": latency
    }

if __name__ == "__main__":
    img = "dataset/sample/test4.png"
    print(infer(img))
