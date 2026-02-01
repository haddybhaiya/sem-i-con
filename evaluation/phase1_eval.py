import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import json
from edge.auto_edge import auto_edge_infer

DATASET = "dataset/1test"   # phase-1 test dataset 
RESULTS = []

for cls in os.listdir(DATASET):
    cls_path = os.path.join(DATASET, cls)
    if not os.path.isdir(cls_path):
        continue

    for img in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img)
        res = auto_edge_infer(img_path)

        RESULTS.append({
            "image": img,
            "true_class": cls,
            "pred_class": res["class"],
            "confidence": res["confidence"],
            "mode": res["mode"],
            "cpu": res["cpu"],
            "latency": res["latency"]
        })

with open("evaluation/report.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("Phase-1 evaluation complete ,report.json generated")
