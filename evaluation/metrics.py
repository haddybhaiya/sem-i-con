import json
from collections import defaultdict 
with open("evaluation/report.json") as f:
    data = json.load(f)
total = len(data)
correct = 0
class_stats = defaultdict(lambda: { "tp":0, "total" : 0 })
latencies = []
for d in data:
    class_stats[d["true_class"]]["total"] += 1
    latencies.append(d["latency"])
    if d["true_class"] == d["pred_class"]:
        correct += 1
        class_stats[d["true_class"]]["tp"] += 1
accuracy = correct / total 
print("\n phase-1 Evaluation Report")
print("total samples:", total)
print("overall accuracy: ", round(accuracy,4))
print("avg latency :", round(sum(latencies)/len(latencies),4),"sec")

print("\n class-wise accuracy:")
for cls,v in class_stats.items():
    if v["total"] > 0:
        acc = v["tp"] / v["total"]  
        print(cls,"+",round(acc,4))