import psutil
import time
import onnxruntime as ort
import numpy as np
import cv2

FP32_MODEL = "models/edge_model.onnx"
INT8_MODEL = "models/edge_model_int8.onnx"

CLASSES = [
    "clean","bridge","cmp","crack","open","ler","via","other"
]

IMG_SIZE_HIGH = 224
IMG_SIZE_LOW = 160

LATENCY_THRESHOLD = 0.15   # seconds
CPU_THRESHOLD = 70         # %

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


def auto_edge_infer(img_path):
    cpu = psutil.cpu_percent(interval=0.1)

    # choose model
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
    out = session.run(None, {input_name: x})
    latency = time.time() - start
    logits = out[0][0]          # raw output
    probs = softmax(logits)     # convert to probabilities
    cls_id = int(np.argmax(probs))

    return {
        "class": CLASSES[cls_id],
        "confidence": float(probs[cls_id]),
        "mode": mode,
        "cpu": cpu,
        "latency": latency
    }

if __name__ == "__main__":
    img = "dataset/sample/test3.png"   # add test image ( i will be using test 3.png)
    result = auto_edge_infer(img)
    print(result)
