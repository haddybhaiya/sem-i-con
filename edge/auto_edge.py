import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import psutil
import time
import onnxruntime as ort
import cv2
import numpy as np

# ---------------- Paths ----------------
FP32_MODEL = "models/convnext_sem.onnx"
FP16_MODEL = "models/convnext_sem_fp16.onnx"

# ---------------- Config ----------------
CLASSES = [
    "clean","bridge","cmp","crack",
    "open","ler","via","other"
]

IMG_SIZE_HIGH = 224
IMG_SIZE_LOW  = 160

CPU_THRESHOLD   = 65
OTHER_THRESHOLD = 0.75

# ---------------- Load sessions ----------------
session_fp32 = ort.InferenceSession(
    FP32_MODEL, providers=["CPUExecutionProvider"]
)

session_fp16 = ort.InferenceSession(
    FP16_MODEL, providers=["CPUExecutionProvider"]
)

input_fp32 = session_fp32.get_inputs()[0].name
input_fp16 = session_fp16.get_inputs()[0].name


# ---------------- Utils ----------------
def preprocess(img_path, size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Invalid image path: {img_path}")

    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0

    # (1, 1, H, W) → ConvNeXt inchans=1
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ---------------- Auto Edge Infer ----------------
def auto_edge_infer(img_path):
    cpu = psutil.cpu_percent(interval=0.1)

    # ---- Model + Resolution selection ----
    if cpu > CPU_THRESHOLD:
        session = session_fp16
        input_name = input_fp16
        size = IMG_SIZE_LOW
        mode = "FP16_LOW_RES"
    else:
        session = session_fp32
        input_name = input_fp32
        size = IMG_SIZE_HIGH
        mode = "FP32_HIGH_RES"

    # ---- Inference ----
    start = time.time()
    x = preprocess(img_path, size)
    logits = session.run(None, {input_name: x})[0][0]
    latency = time.time() - start

    probs = softmax(logits)
    cls_id = int(np.argmax(probs))
    confidence = float(probs[cls_id])
    raw_class = CLASSES[cls_id]

    pred_class = raw_class

    # -------- Semantic Guard --------
    if confidence < OTHER_THRESHOLD:
        pred_class = "other"

    if raw_class in ["bridge", "crack", "open"] and confidence < 0.85:
        pred_class = "other"

    if raw_class == "cmp" and confidence < 0.80:
        pred_class = "other"
    # --------------------------------

    return {
        "class": pred_class,
        "raw": raw_class,
        "confidence": confidence,
        "cpu": cpu,
        "latency": latency,
        "mode": mode
    }


# ---------------- Test ----------------
if __name__ == "__main__":
    img = "dataset/sample/test4.png"
    print(auto_edge_infer(img))
