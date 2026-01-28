import timm
import torch.nn as nn

def build_model(num_classes=8):
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=num_classes
    )
    return model
