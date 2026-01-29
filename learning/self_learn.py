import os
import csv
import shutil
from datetime import datetime

# Paths
BUFFER_DIR = "learning/buffer"
OTHER_DIR = os.path.join(BUFFER_DIR, "other")
LOWCONF_DIR = os.path.join(BUFFER_DIR, "low_confidence")
LOG_FILE = os.path.join(BUFFER_DIR, "logs.csv")

CONF_THRESHOLD = 0.75   # below this = learning sample

os.makedirs(OTHER_DIR, exist_ok=True)
os.makedirs(LOWCONF_DIR, exist_ok=True)
os.makedirs(BUFFER_DIR, exist_ok=True)

# Init log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image", "predicted_class", "confidence", "reason"])

def log_sample(img_path, pred_class, confidence):
    ts = datetime.now().isoformat()
    reason = "other_class" if pred_class == "other" else "low_confidence"

    # destination
    if pred_class == "other":
        dst_dir = OTHER_DIR
    else:
        dst_dir = LOWCONF_DIR

    fname = os.path.basename(img_path)
    new_name = f"{ts.replace(':','_')}_{fname}"
    dst_path = os.path.join(dst_dir, new_name)

    shutil.copy(img_path, dst_path)

    # log
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, img_path, pred_class, confidence, reason])

    print(f"[SELF-LEARN] stored -> {dst_path}")

def self_learn_hook(img_path, pred_class, confidence):
    if pred_class == "other" or confidence < CONF_THRESHOLD:
        log_sample(img_path, pred_class, confidence)
