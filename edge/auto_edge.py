import psutil
import time
import onnxruntime as ort
import numpy as np
import cv2

FP32_MODEL = "models/1edge_model.onnx"
INT8_MODEL = "models/1edge_model_int8.onnx"

CLASSES = [
    "clean","bridge","cmp","crack","open","ler","via","other"
]

IMG_SIZE_HIGH = 224
IMG_SIZE_LOW = 160

CPU_THRESHOLD = 70

CONF_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.18


def load_session(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


session_fp32 = load_session(FP32_MODEL)
session_int8 = load_session(INT8_MODEL)

input_fp32 = session_fp32.get_inputs()[0].name
input_int8 = session_int8.get_inputs()[0].name


def preprocess(img_path, size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


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

    top2 = probs.argsort()[-2:][::-1]
    top1, top2_id = top2[0], top2[1]

    confidence = float(probs[top1])
    margin = confidence - float(probs[top2_id])

    pred_class = CLASSES[top1]

    if confidence < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
        pred_class = "other"

    return {
        "class": pred_class,
        "confidence": confidence,
        "margin": margin,
        "raw_top1": CLASSES[top1],
        "raw_top2": CLASSES[top2_id],
        "mode": mode,
        "cpu": cpu,
        "latency": latency
    }


if __name__ == "__main__":
    img = "dataset/sample/test3.png"
    print(auto_edge_infer(img))
