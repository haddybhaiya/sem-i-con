import psutil
import time
import onnxruntime as ort
import numpy as np
import cv2

FP32_MODEL = "models/1edge_model.onnx"
INT8_MODEL = "models/1edge_model_int8.onnx"

CLASSES = ["clean","bridge","cmp","crack","open","ler","via","other"]

IMG_SIZE_HIGH = 224
IMG_SIZE_LOW = 160
CPU_THRESHOLD = 70  # %

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


def load_session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


session_fp32 = load_session(FP32_MODEL)
session_int8 = load_session(INT8_MODEL)

input_fp32 = session_fp32.get_inputs()[0].name
input_int8 = session_int8.get_inputs()[0].name


def preprocess(img_path, size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def auto_edge_infer(img_path):
    cpu = psutil.cpu_percent(interval=0.1)

    if cpu > CPU_THRESHOLD:
        session = session_int8
        input_name = input_int8
        size = IMG_SIZE_LOW
        mode = "INT8_LOW_RES"
    else:
        session = session_fp32
        input_name = input_fp32
        size = IMG_SIZE_HIGH
        mode = "FP32_HIGH_RES"

    start = time.time()
    x = preprocess(img_path, size)
    logits = session.run(None, {input_name: x})[0][0]
    latency = time.time() - start

    probs = softmax(logits)
    top2 = np.argsort(probs)[-2:][::-1]

    raw_class = CLASSES[top2[0]]
    confidence = float(probs[top2[0]])
    pred_class = raw_class

    # ---- class-wise threshold ----
    if confidence < CLASS_THRESHOLDS[raw_class]:
        pred_class = "other"

    # ---- ambiguity resolver ----
    second_class = CLASSES[top2[1]]
    if raw_class in ["cmp", "bridge", "open"] and second_class in ["via", "crack"]:
        if abs(probs[top2[0]] - probs[top2[1]]) < 0.08:
            pred_class = "other"

    return {
        "class": pred_class,
        "raw": raw_class,
        "confidence": confidence,
        "top2": [(CLASSES[i], float(probs[i])) for i in top2],
        "mode": mode,
        "cpu": cpu,
        "latency": latency
    }


if __name__ == "__main__":
    img = "dataset/sample/test6.png"
    print(auto_edge_infer(img))
