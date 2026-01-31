import json
from collections import Counter

with open("evaluation/report.json") as f:
    data = json.load(f)

modes = Counter([d["mode"] for d in data])
lat = [d["latency"] for d in data]
cpu = [d["cpu"] for d in data]

print("\nEdge Benchmark Report")
print("Modes:", modes)
print("Avg Latency:", sum(lat)/len(lat))
print("Avg CPU:", sum(cpu)/len(cpu))
print("Max Latency:", max(lat))
print("Min Latency:", min(lat))
