import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from typing import Any

def swin_tiny_custom(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> nn.Module:
    """
    Swin Transformer Tiny Custom Model.
    If 'pretrained=True', loads ImageNet weights.
    """
    weights = Swin_T_Weights.DEFAULT if pretrained else None
    model = swin_t(weights=weights, progress=progress, **kwargs)
    return model
