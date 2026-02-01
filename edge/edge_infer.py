import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import onnxruntime as ort
import cv2
import numpy as np

MODEL_PATH = "models/1edge_model_int8.onnx"

CLASSES = [
    "clean","bridge","cmp","crack","open","ler","via","other"
]

IMG_SIZE = 224

# ---- Tuned for your dataset ----
CONF_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.18   # top1 - top2 confidence gap
# --------------------------------

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def infer(img_path):
    x = preprocess(img_path)
    logits = session.run(None, {input_name: x})[0][0]
    probs = softmax(logits)

    top2 = probs.argsort()[-2:][::-1]
    top1, top2_id = top2[0], top2[1]

    confidence = float(probs[top1])
    margin = confidence - float(probs[top2_id])

    pred_class = CLASSES[top1]

    # --------- Rejection logic ----------
    if confidence < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
        pred_class = "other"
    # -----------------------------------

    return {
        "class": pred_class,
        "confidence": confidence,
        "margin": margin,
        "raw_top1": CLASSES[top1],
        "raw_top2": CLASSES[top2_id]
    }


if __name__ == "__main__":
    img = "dataset/sample/test7.jpg"
    print(infer(img))
