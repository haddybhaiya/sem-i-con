# learning/self_learn.py

import os
import json
import shutil
from datetime import datetime

SELF_LEARN_DIR = "self_learn_buffer"
LOG_FILE = os.path.join(SELF_LEARN_DIR, "log.json")

CONFIDENCE_TRIGGER = 0.70   # below this → suspicious

os.makedirs(SELF_LEARN_DIR, exist_ok=True)

def self_learn_hook(img_path, pred_class, confidence):
    """
    Collects low-confidence samples for future retraining
    """
    if confidence >= CONFIDENCE_TRIGGER:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = os.path.basename(img_path)

    sample_id = f"{timestamp}_{img_name}"
    save_path = os.path.join(SELF_LEARN_DIR, sample_id)

    shutil.copy(img_path, save_path)

    record = {
        "image": sample_id,
        "predicted_class": pred_class,
        "confidence": confidence,
        "time": timestamp
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
