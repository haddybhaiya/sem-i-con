# training/model.py

import timm
import torch.nn as nn

def build_model(num_classes=8):
    """
    MobileNetV3 distilled student model
    SEM compatible: grayscale input
    """
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=False,
        in_chans=1,          # 🔴 SEM grayscale
        num_classes=num_classes
    )
    return model
