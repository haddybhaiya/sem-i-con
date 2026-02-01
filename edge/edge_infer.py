import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import onnxruntime as ort
import cv2
import numpy as np
from learning.self_learn import self_learn_hook


MODEL_PATH = "models/1edge_model_int8.onnx"
test_img = "dataset/1test/open/open_171.png"  
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
OTHER_THRESHOLD = 0.80

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

IMG_SIZE = 224

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

def entropy(p):
    return -np.sum(p * np.log(p + 1e-9))


def infer(img_path):
    x = preprocess(img_path)
    outputs = session.run(None, {input_name: x})

    logits = outputs[0][0]
    probs = softmax(logits)

    # top-1 and top-2
    sorted_idx = np.argsort(probs)[::-1]
    top1, top2 = sorted_idx[0], sorted_idx[1]

    top1_class = CLASSES[top1]
    top1_conf = float(probs[top1])
    margin = float(probs[top1] - probs[top2])
    ent = entropy(probs)

    final_class = top1_class

    # -------- decision logic -------- #

    #  Very uncertain → other
    if top1_conf < 0.60:
        final_class = "other"

    #  Ambiguous boundary → other
    elif margin < 0.15:
        final_class = "other"

    #  High confusion → other
    elif ent > 1.6:
        final_class = "other"

    #  Structural defect sanity
    elif top1_class in ["bridge", "crack", "open"] and top1_conf < 0.85:
        final_class = "other"

    # CMP stricter gate
    elif top1_class == "cmp" and top1_conf < 0.80:
        final_class = "other"

    # self-learning hook
    self_learn_hook(img_path, final_class, top1_conf)

    return {
        "raw_class": top1_class,
        "final_class": final_class,
        "confidence": top1_conf,
        "margin": margin,
        "entropy": ent
    }

if __name__ == "__main__":
    result = infer(test_img)
    print("Raw Prediction:", result["raw_class"])
    print("Final Prediction:", result["final_class"])
    print("Confidence:", result["confidence"])
    print("Margin:", result["margin"])
    print("Entropy:", result["entropy"])    
