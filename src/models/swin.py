import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

def get_model(num_classes=7):
    model = swin_t(weights=Swin_T_Weights.DEFAULT)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(2, 3, 224, 224)
    print("Output shape:", model(x).shape)