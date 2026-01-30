import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from edge.auto_edge import auto_edge_infer


TEST_DIR = "dataset/sample"

results = []

for f in os.listdir(TEST_DIR):
    if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")):
        path = os.path.join(TEST_DIR, f)
        out = auto_edge_infer(path)
        out["image"] = f
        results.append(out)
        print(f"{f} -> {out}")

print("\nSummary of Results:")
for r in results:
    print(r)
