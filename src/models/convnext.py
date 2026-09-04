import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def get_model(num_classes=7):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
    )
    return model

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(2, 3, 224, 224)
    print("Output shape:", model(x).shape)