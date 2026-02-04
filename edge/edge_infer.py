import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import onnxruntime as ort
import cv2
import numpy as np
from learning.self_learn import self_learn_hook

MODEL_PATH = "models/convnext_sem.onnx"

CLASSES = [
    "clean", "bridge", "cmp", "crack",
    "open", "ler", "via", "other"
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
        raise ValueError(f"Could not read image {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    # (1,1,H,W) — true grayscale
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def infer(img_path):
    x = preprocess(img_path)
    logits = session.run(None, {input_name: x})[0][0]
    probs = softmax(logits)

    cls_id = int(np.argmax(probs))
    confidence = float(probs[cls_id])
    raw_class = CLASSES[cls_id]

    pred_class = raw_class

    # ---------- Semantic Guard ----------
    if confidence < OTHER_THRESHOLD:
        pred_class = "other"

    if raw_class in ["bridge", "crack", "open"] and confidence < 0.85:
        pred_class = "other"

    if raw_class == "cmp" and confidence < 0.80:
        pred_class = "other"
    # -----------------------------------

    self_learn_hook(img_path, pred_class, confidence)
    return pred_class, confidence, raw_class


if __name__ == "__main__":
    img = "dataset/sample/test1.png"
    label, conf, raw = infer(img)
    print("Raw:", raw)
    print("Final:", label)
    print("Confidence:", conf)
