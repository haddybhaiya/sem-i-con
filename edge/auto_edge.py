import sys, os
import psutil
import time
import onnxruntime as ort
import numpy as np
import cv2

# ----------------- Path fix -----------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# --------------------------------------------

FP32_MODEL = "models/1edge_model.onnx"
INT8_MODEL = "models/1edge_model_int8.onnx"

CLASSES = [
    "clean",
    "bridge",
    "cmp",
    "crack",
    "open",
    "ler",
    "via",
    "other"
]

IMG_SIZE_HIGH = 224
IMG_SIZE_LOW = 160

CPU_THRESHOLD = 70  # %



def load_session(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


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


def entropy(p):
    return -np.sum(p * np.log(p + 1e-9))


def auto_edge_infer(img_path):
    cpu = psutil.cpu_percent(interval=0.1)

    # -------- Auto Edge Selection --------
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
    # ------------------------------------

    start = time.time()
    x = preprocess(img_path, size)
    outputs = session.run(None, {input_name: x})
    latency = time.time() - start

    logits = outputs[0][0]
    probs = softmax(logits)

    # -------- Decision Logic --------
    sorted_idx = np.argsort(probs)[::-1]
    top1, top2 = sorted_idx[0], sorted_idx[1]

    top1_class = CLASSES[top1]
    top1_conf = float(probs[top1])
    margin = float(probs[top1] - probs[top2])
    ent = entropy(probs)

    final_class = top1_class

    # Low confidence
    if top1_conf < 0.60:
        final_class = "other"

    #  Ambiguous boundary
    elif margin < 0.15:
        final_class = "other"

    #  High uncertainty
    elif ent > 1.6:
        final_class = "other"

    #  Structural defect sanity
    elif top1_class in ["bridge", "crack", "open"] and top1_conf < 0.85:
        final_class = "other"

    #  CMP stricter gate
    elif top1_class == "cmp" and top1_conf < 0.80:
        final_class = "other"
    # ------------------------------------

    return {
        "class": final_class,
        "raw_class": top1_class,
        "confidence": top1_conf,
        "margin": margin,
        "entropy": ent,
        "mode": mode,
        "cpu": cpu,
        "latency": latency
    }


if __name__ == "__main__":
    img = "dataset/sample/test3.png"
    result = auto_edge_infer(img)
    print(result)
