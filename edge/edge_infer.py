import sys, os
import numpy as np
import cv2
import onnxruntime as ort

# project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from learning.self_learn import self_learn_hook

MODEL_PATH = "models/1edge_model_int8.onnx"
TEST_IMG = "dataset/sample/test7.jpg"

CLASSES = ["clean","bridge","cmp","crack","open","ler","via","other"]

# ---- calibrated thresholds (IMPORTANT) ----
CLASS_THRESHOLDS = {
    "clean": 0.40,
    "cmp": 0.45,
    "crack": 0.50,
    "open": 0.55,
    "ler": 0.45,
    "bridge": 0.60,
    "via": 0.55,
    "other": 0.70
}

IMG_SIZE = 224

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def infer(img_path):
    x = preprocess(img_path)
    logits = session.run(None, {input_name: x})[0][0]
    probs = softmax(logits)

    # top predictions
    top2 = np.argsort(probs)[-2:][::-1]
    raw_class = CLASSES[top2[0]]
    confidence = float(probs[top2[0]])

    pred_class = raw_class

    # ---- class-wise thresholding ----
    if confidence < CLASS_THRESHOLDS[raw_class]:
        pred_class = "other"

    # ---- ambiguity resolver (ONLY ONE RULE) ----
    second_class = CLASSES[top2[1]]
    if raw_class in ["cmp", "bridge", "open"] and second_class in ["via", "crack"]:
        if abs(probs[top2[0]] - probs[top2[1]]) < 0.08:
            pred_class = "other"

    self_learn_hook(img_path, pred_class, confidence)

    return {
        "final": pred_class,
        "raw": raw_class,
        "confidence": confidence,
        "top2": [(CLASSES[i], float(probs[i])) for i in top2]
    }


if __name__ == "__main__":
    out = infer(TEST_IMG)
    print(out)
