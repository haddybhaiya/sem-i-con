import torch
import torch.nn as nn
import timm


def build_model(
    num_classes: int = 8,
    in_chans: int = 1,
    pretrained: bool = False
):
    """
    ConvNeXt-Tiny backbone adapted for SEM grayscale images
    """

    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans
    )

    return model
