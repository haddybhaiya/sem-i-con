import json
from collections import defaultdict

with open("evaluation/report.json") as f:
    data = json.load(f)

total = len(data)
correct = 0
class_stats = defaultdict(lambda: {"tp": 0, "total": 0})
latencies = []

for d in data:
    true = d["true_class"]
    pred = d["pred_class"]

    class_stats[true]["total"] += 1
    latencies.append(d["latency"])

    if true == pred:
        correct += 1
        class_stats[true]["tp"] += 1

accuracy = correct / total if total > 0 else 0
avg_latency = sum(latencies) / len(latencies) if latencies else 0

print("\nPhase-1 Evaluation Report")
print("Total samples:", total)
print("Overall accuracy:", round(accuracy, 4))
print("Avg latency:", round(avg_latency, 4), "sec")

print("\nClass-wise accuracy:")
for cls, v in class_stats.items():
    acc = v["tp"] / v["total"] if v["total"] > 0 else 0
    print(cls, ":", round(acc, 4))
