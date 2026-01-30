import onnxruntime as ort
import cv2
import numpy as np
from learning.self_learn import self_learn_hook


MODEL_PATH = "models/edge_model_int8.onnx"
test_img = "dataset/sample/test3.png"  
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

def infer(img_path):
    x = preprocess(img_path)
    outputs = session.run(None, {input_name: x})

    logits = outputs[0][0]          # raw model output
    probs = softmax(logits)         # apply softmax for probabilities

    cls_id = int(np.argmax(probs))
    raw_class = CLASSES[cls_id]
    confidence = float(probs[cls_id])

    pred_class = raw_class
    #fix for irregular classification cases
    # Low confidence → other
    if confidence < OTHER_THRESHOLD:
        pred_class = "other"

    # Line-structure ambiguity correction
    if pred_class in ["bridge", "crack", "open"] and confidence < 0.90:
        pred_class = "other"

    # CMP false-positive correction
    if pred_class == "cmp" and confidence < 0.85:
        pred_class = "other"
    # self-learning trigger
    self_learn_hook(img_path, pred_class, confidence)

    return pred_class, confidence, raw_class

if __name__ == "__main__":
    label, conf, raw = infer(test_img)
    print("Raw Prediction:", raw)
    print("Final Prediction:", label)
    print("Confidence:", conf)


