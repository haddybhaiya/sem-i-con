import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from edge.auto_edge import auto_edge_infer
from collections import defaultdict
import os

TEST_DATASET = "dataset/test"

CLASSES = [
    "clean","bridge","cmp","crack",
    "open","ler","via","other"
]

correct = defaultdict(int)
total = defaultdict(int)

for cls in CLASSES:
    cls_path = os.path.join(TEST_DATASET, cls)
    if not os.path.exists(cls_path):
        continue

    for img in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img)
        result = auto_edge_infer(img_path)

        total[cls] += 1
        if result["class"] == cls:
            correct[cls] += 1

print("\nClass-wise accuracy:")
for cls in CLASSES:
    if total[cls] == 0:
        continue
    acc = correct[cls] / total[cls]
    print(f"{cls:>8} : {acc:.3f}")
overall_acc = sum(correct.values()) / sum(total.values())
print(f"\nOverall accuracy: {overall_acc:.3f}")