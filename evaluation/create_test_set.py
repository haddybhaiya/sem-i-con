import os
import shutil
import random

SRC = "dataset/1synthetic_dataset"
DST = "dataset/1test"
SAMPLES_PER_CLASS = 25   #(small + fair)
SEED = 42

random.seed(SEED)
os.makedirs(DST, exist_ok=True)

for cls in os.listdir(SRC):
    src_cls = os.path.join(SRC, cls)
    dst_cls = os.path.join(DST, cls)

    if not os.path.isdir(src_cls):
        continue

    os.makedirs(dst_cls, exist_ok=True)

    images = [
        f for f in os.listdir(src_cls)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(images) < SAMPLES_PER_CLASS:
        print(f"[WARN] {cls} has only {len(images)} images")
        sample = images
    else:
        sample = random.sample(images, SAMPLES_PER_CLASS)

    for img in sample:
        shutil.copy(
            os.path.join(src_cls, img),
            os.path.join(dst_cls, img)
        )

    print(f"{cls}: {len(sample)} test images created")

print("\nTest dataset created successfully")
